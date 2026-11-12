"""Hugging Face Transformers とDINOv3のpatch特徴量抽出を担う。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from dinov3_pipeline.common.constants import (
    FEATURE_DTYPE,
    HF_MODEL_IDS,
    MASK_FOREGROUND,
    PATCH_TOKENS_KEY,
    PRETRAIN_NORMALIZATION,
)
from dinov3_pipeline.common.models import DataContractError, FeatureArtifact, ReadySample
from dinov3_pipeline.data_loading.preprocessing import prepare_crop


def patch_tokens_to_hwc(patch_tokens: np.ndarray, grid_height: int, grid_width: int) -> np.ndarray:
    """2D の patch token を HWC の空間マップに再配置する。"""
    if grid_height <= 0 or grid_width <= 0 or patch_tokens.ndim != 2:
        raise DataContractError("patch tokenには空でない2次元行列と正のgrid寸法が必要である")
    if patch_tokens.shape[0] != grid_height * grid_width or patch_tokens.shape[1] <= 0:
        raise DataContractError("patch token数が指定された空間gridと一致しない")
    return np.asarray(patch_tokens.reshape(grid_height, grid_width, -1), dtype=np.float32)


class Dinov3FeatureExtractor:
    """DINOv3 の patch token を抽出して、HWC の特徴量として保存する。"""

    def __init__(
        self,
        model_name: str,
        pretrain_dataset: str,
        crop_size: int,
        device_name: str | None,
        mask_rgb_background: bool = True,
    ) -> None:
        """公式モデルIDとdeviceを解決し、Transformersで読み込む。"""
        if model_name not in HF_MODEL_IDS:
            raise ValueError(f"model_nameは公式Hugging Face IDのいずれかである必要がある: {model_name}")

        self._torch = torch
        self.model_name = model_name
        self.model_id = model_name
        self.crop_size = crop_size
        self.mask_rgb_background = mask_rgb_background
        self._mean_std = PRETRAIN_NORMALIZATION[pretrain_dataset]
        self.device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        from_pretrained_kwargs = {"token": hf_token} if hf_token else {}

        try:
            model = AutoModel.from_pretrained(self.model_id, **from_pretrained_kwargs)
        except Exception as error:
            raise RuntimeError(f"DINOv3を読み込めない: {self.model_id}（{error}）") from error
        self.model = model.to(self.device)
        self.model.eval()
        self.patch_size = int(getattr(self.model.config, "patch_size", 16))
        self.num_layers = int(getattr(self.model.config, "num_hidden_layers", 0) or getattr(self.model.config, "num_layers", 0) or 0)
        if crop_size <= 0 or crop_size % self.patch_size:
            raise ValueError(f"crop_sizeは正の値で、patch size {self.patch_size}で割り切れる必要がある")
        if self.num_layers <= 0:
            raise ValueError(f"モデルのlayer数が検出できない: {self.model_id}")

    def extract(self, sample: ReadySample, destination: Path) -> FeatureArtifact:
        """1 sample を前処理し、最終層および中間層の特徴量を保存する。"""
        cropped = prepare_crop(sample, self.crop_size, mask_rgb_background=self.mask_rgb_background)
        pixels = np.asarray(cropped.rgb, dtype=np.float32) / MASK_FOREGROUND
        tensor = self._torch.from_numpy(pixels).permute(2, 0, 1)
        mean = self._torch.tensor(self._mean_std[0], dtype=tensor.dtype).view(3, 1, 1)
        std = self._torch.tensor(self._mean_std[1], dtype=tensor.dtype).view(3, 1, 1)
        mask = self._torch.from_numpy(np.asarray(cropped.mask, dtype=np.float32) / MASK_FOREGROUND)
        normalized = (tensor - mean) / std
        if self.mask_rgb_background:
            normalized = normalized * mask.unsqueeze(0)
        batch = normalized.unsqueeze(0).to(self.device)

        with self._torch.inference_mode():
            grid = self.crop_size // self.patch_size
            block_tokens, final_tokens = self._intermediate_layers(batch, grid)

        block_features: list[np.ndarray] = []
        for token_map in block_tokens:
            block_features.append(patch_tokens_to_hwc(token_map.float().cpu().numpy(), grid, grid))

        features = patch_tokens_to_hwc(final_tokens.float().cpu().numpy(), grid, grid)
        self._save_array(destination, features)
        block_paths = []
        for index, block_features_array in enumerate(block_features, start=1):
            block_path = destination.parent / f"feature_block_{index}" / destination.name
            self._save_array(block_path, block_features_array)
            block_paths.append(block_path)
        return FeatureArtifact(
            path=destination,
            shape=(int(features.shape[0]), int(features.shape[1]), int(features.shape[2])),
            dtype=str(features.dtype),
            model_name=self.model_name,
            block_paths=tuple(block_paths),
        )

    def _intermediate_layers(self, images: object, grid: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        outputs = self.model(images, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.get("hidden_states")
        final_hidden_state = outputs.get("last_hidden_state")
        if hidden_states is None or final_hidden_state is None:
            raise DataContractError("Transformersのhidden_statesとlast_hidden_stateが必要である")
        if len(hidden_states) != self.num_layers + 1:
            raise DataContractError(f"hidden_statesはembedding入力と{self.num_layers}個のBlock出力が必要だが、{len(hidden_states)}個である")
        selected = [self._patch_tokens(hidden_states[index], grid) for index in self._layer_indices()]
        return selected, self._patch_tokens(final_hidden_state, grid)

    def _patch_tokens(self, output: torch.Tensor, grid: int) -> torch.Tensor:
        if output.ndim == 3:
            output = output[0]
        if output.ndim != 2:
            raise DataContractError("中間層出力はpatch token行列または空間patch mapである必要がある")

        patch_count = grid * grid
        token_count = output.shape[0]
        special_token_count = self._special_token_count()
        if token_count != special_token_count + patch_count:
            raise DataContractError(f"special token数を除いたpatch tokenは{patch_count}個必要だが、出力{token_count}個、モデルのspecial token数{special_token_count}個である")
        return output[special_token_count:]

    def _special_token_count(self) -> int:
        embeddings = getattr(self.model, "embeddings", None)
        cls_token = getattr(embeddings, "cls_token", None)
        register_tokens = getattr(embeddings, "register_tokens", None)
        if cls_token is None or register_tokens is None or cls_token.ndim < 2 or register_tokens.ndim < 2:
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
        layers = self.num_layers
        return [layers // 4, 2 * layers // 4, 3 * layers // 4, layers]

    @staticmethod
    def _save_array(destination: Path, features: np.ndarray) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=destination.suffix, delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, features, allow_pickle=False)
        temporary.replace(destination)
