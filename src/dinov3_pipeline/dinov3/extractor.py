"""Hugging Face Transformers とDINOv3のpatch特徴量抽出を担う。"""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, PreTrainedModel

from dinov3_pipeline.common.constants import (
    HF_MODEL_IDS,
    MASK_FOREGROUND,
    PRETRAIN_NORMALIZATION,
)
from dinov3_pipeline.common.models import DataContractError

PATCH_TOKEN_MATRIX_DIMS = 2
BATCHED_PATCH_TOKEN_DIMS = 3
RGB_CHANNEL_COUNT = 3


def patch_tokens_to_hwc(patch_tokens: np.ndarray, grid_height: int, grid_width: int) -> np.ndarray:
    """2D patch tokenをHWCの空間マップへ再配置する。

    Args:
        patch_tokens: patchごとの特徴量を並べた行列。
        grid_height: 出力特徴マップの高さ。
        grid_width: 出力特徴マップの幅。

    Returns:
        float32のHWC特徴マップ。

    Raises:
        DataContractError: grid寸法またはpatch tokenのshapeが不正な場合。
    """
    if grid_height <= 0 or grid_width <= 0 or patch_tokens.ndim != PATCH_TOKEN_MATRIX_DIMS:
        raise DataContractError("patch tokenには空でない2次元行列と正のgrid寸法が必要である")
    if patch_tokens.shape[0] != grid_height * grid_width or patch_tokens.shape[1] <= 0:
        raise DataContractError("patch token数が指定された空間gridと一致しない")
    return np.asarray(patch_tokens.reshape(grid_height, grid_width, -1), dtype=np.float32)


class Dinov3FeatureExtractor:
    """整形済み画像からDINOv3のPatch Token特徴量を推論する。"""

    def __init__(
        self,
        model_name: str,
        pretrain_dataset: str,
        device_name: str | None,
    ) -> None:
        """指定された公式モデルを読み込み、推論設定を準備する。

        Args:
            model_name: Hugging Faceの公式DINOv3モデルID。
            pretrain_dataset: 正規化統計を選ぶ事前学習データ名。
            device_name: 使用device名。省略時はCUDA利用可否から選ぶ。

        Raises:
            ValueError: モデルIDまたはモデル構成が不正な場合。
            RuntimeError: Transformersからモデルを読み込めない場合。
        """
        if model_name not in HF_MODEL_IDS:
            raise ValueError(f"model_nameは公式Hugging Face IDのいずれかである必要がある: {model_name}")

        self._torch = torch
        self.model_name = model_name
        self.model_id = model_name
        self._mean_std = PRETRAIN_NORMALIZATION[pretrain_dataset]
        self.device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model: PreTrainedModel = self._load_model()
        self.model.eval()
        self.patch_size = int(getattr(self.model.config, "patch_size", 16))
        configured_layers = getattr(self.model.config, "num_hidden_layers", 0)
        if not configured_layers:
            configured_layers = getattr(self.model.config, "num_layers", 0)
        self.num_layers = int(configured_layers or 0)
        if self.num_layers <= 0:
            raise ValueError(f"モデルのlayer数が検出できない: {self.model_id}")

    def _load_model(self) -> PreTrainedModel:
        """指定deviceにDINOv3モデルをロードする。

        Returns:
            評価modeへ切替可能なTransformersモデル。

        Raises:
            RuntimeError: Transformersからモデルを読み込めない場合。
        """
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        from_pretrained_kwargs = {"token": hf_token} if hf_token else {}

        try:
            model = AutoModel.from_pretrained(self.model_id, **from_pretrained_kwargs)
        except Exception as error:
            raise RuntimeError(f"DINOv3を読み込めない: {self.model_id}（{error}）") from error
        return model.to(self.device)

    def infer(
        self,
        rgb_image: Image.Image,
        mask_image: Image.Image | None = None,
    ) -> dict[str, np.ndarray]:
        """整形済みRGB画像を推論し、最終・中間層のHWC特徴量を返す。

        Args:
            rgb_image: RGB modeで、crop等の整形を済ませた画像。
            mask_image: RGB背景maskを推論入力にも適用する場合の整形済みmask。

        Returns:
            最終層と4つの中間層のfloat32 HWC特徴量。

        Raises:
            DataContractError: 入力画像またはTransformer出力が契約に合わない場合。
        """
        if rgb_image.mode != "RGB":
            raise DataContractError("rgb_imageはRGB modeである必要がある")
        image_width, image_height = rgb_image.size
        if image_width <= 0 or image_height <= 0:
            raise DataContractError("rgb_imageの寸法は正である必要がある")
        if image_width % self.patch_size or image_height % self.patch_size:
            raise DataContractError(f"rgb_imageの縦横はpatch size {self.patch_size}で割り切れる必要がある")
        if mask_image is not None and mask_image.size != rgb_image.size:
            raise DataContractError("mask_imageとrgb_imageの寸法が一致する必要がある")

        pixels = np.asarray(rgb_image, dtype=np.float32) / MASK_FOREGROUND
        tensor = self._torch.from_numpy(pixels).permute(2, 0, 1)
        mean = self._torch.tensor(self._mean_std[0], dtype=tensor.dtype).view(RGB_CHANNEL_COUNT, 1, 1)
        std = self._torch.tensor(self._mean_std[1], dtype=tensor.dtype).view(RGB_CHANNEL_COUNT, 1, 1)
        normalized = (tensor - mean) / std
        if mask_image is not None:
            mask = self._torch.from_numpy(np.asarray(mask_image.convert("L"), dtype=np.float32) / MASK_FOREGROUND)
            normalized = normalized * mask.unsqueeze(0)
        batch = normalized.unsqueeze(0).to(self.device)

        with self._torch.inference_mode():
            grid_height = image_height // self.patch_size
            grid_width = image_width // self.patch_size
            block_tokens, final_tokens = self._intermediate_layers(batch, grid_height, grid_width)

        # 学習メモ: 辞書内包表記は各要素から辞書を作る。for文で順に追加する書き方を短く表す。
        features = {f"feature_block_{index}": patch_tokens_to_hwc(token_map.float().cpu().numpy(), grid_height, grid_width) for index, token_map in enumerate(block_tokens, start=1)}
        features["feature"] = patch_tokens_to_hwc(final_tokens.float().cpu().numpy(), grid_height, grid_width)
        return features

    def _intermediate_layers(self, images: object, grid_height: int, grid_width: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Transformerから選択中間層と最終層のpatch tokenを取り出す。"""
        outputs = self.model(images, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.get("hidden_states")
        final_hidden_state = outputs.get("last_hidden_state")
        if hidden_states is None or final_hidden_state is None:
            raise DataContractError("Transformersのhidden_statesとlast_hidden_stateが必要である")
        if len(hidden_states) != self.num_layers + 1:
            raise DataContractError(f"hidden_statesはembedding入力と{self.num_layers}個のBlock出力が必要だが、{len(hidden_states)}個である")
        selected = [self._patch_tokens(hidden_states[index], grid_height, grid_width) for index in self._layer_indices()]
        return selected, self._patch_tokens(final_hidden_state, grid_height, grid_width)

    def _patch_tokens(self, output: torch.Tensor, grid_height: int, grid_width: int) -> torch.Tensor:
        if output.ndim == BATCHED_PATCH_TOKEN_DIMS:
            output = output[0]
        if output.ndim != PATCH_TOKEN_MATRIX_DIMS:
            raise DataContractError("中間層出力はpatch token行列または空間patch mapである必要がある")

        patch_count = grid_height * grid_width
        token_count = output.shape[0]
        special_token_count = self._special_token_count()
        if token_count != special_token_count + patch_count:
            raise DataContractError(f"special token数を除いたpatch tokenは{patch_count}個必要だが、出力{token_count}個、モデルのspecial token数{special_token_count}個である")
        return output[special_token_count:]

    def _special_token_count(self) -> int:
        """モデル埋め込み情報からCLSとregister tokenの個数を返す。"""
        embeddings = getattr(self.model, "embeddings", None)
        cls_token = getattr(embeddings, "cls_token", None)
        register_tokens = getattr(embeddings, "register_tokens", None)
        if cls_token is None or register_tokens is None or cls_token.ndim < PATCH_TOKEN_MATRIX_DIMS or register_tokens.ndim < PATCH_TOKEN_MATRIX_DIMS:
            raise DataContractError("モデル出力のspecial token境界をモデル情報から識別できない")

        cls_count = int(cls_token.shape[-2])
        register_count = int(register_tokens.shape[-2])
        configured_register_count = getattr(self.model.config, "num_register_tokens", None)
        if cls_count <= 0 or register_count < 0:
            raise DataContractError("モデルのspecial token数が不正である")
        if configured_register_count is not None and register_count != int(configured_register_count):
            raise DataContractError("モデル設定と埋め込み層のregister token数が一致しない")
        return cls_count + register_count

    def _layer_indices(self) -> list[int]:
        """モデル深さに応じた4つの中間層位置を返す。"""
        layers = self.num_layers
        return [layers // 4, 2 * layers // 4, 3 * layers // 4, layers]
