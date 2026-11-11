"""Hugging Face Transformers とDINOv3のpatch特徴量抽出を担う。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from dinov3_pipeline.common.constants import (
    FEATURE_DTYPE,
    LEGACY_MODEL_ALIASES,
    MASK_FOREGROUND,
    PATCH_TOKENS_KEY,
    PRETRAIN_NORMALIZATION,
)
from dinov3_pipeline.common.file_hash import sha256_of_file
from dinov3_pipeline.common.models import DataContractError, FeatureArtifact, ReadySample
from dinov3_pipeline.data_loading.preprocessing import prepare_crop


def _canonical_model_name(model_name: str) -> str:
    return LEGACY_MODEL_ALIASES.get(model_name, model_name)


def patch_tokens_to_hwc(patch_tokens: np.ndarray, grid_height: int, grid_width: int) -> np.ndarray:
    if grid_height <= 0 or grid_width <= 0 or patch_tokens.ndim != 2:
        raise DataContractError("patch tokenには空でない2次元行列と正のgrid寸法が必要である")
    if patch_tokens.shape[0] != grid_height * grid_width or patch_tokens.shape[1] <= 0:
        raise DataContractError("patch token数が指定された空間gridと一致しない")
    return np.asarray(patch_tokens.reshape(grid_height, grid_width, -1), dtype=np.float32)


class Dinov3FeatureExtractor:
    def __init__(
        self,
        repo_dir: Path | None,
        model_name: str,
        weights_path: Path | None,
        pretrain_dataset: str,
        crop_size: int,
        device_name: str | None,
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("DINOv3の推論にはtorchが必要である") from error
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise RuntimeError("DINOv3の推論にはtransformersが必要である") from error

        normalized_model_name = _canonical_model_name(model_name)
        if repo_dir is not None and str(repo_dir) and not Path(repo_dir).exists() and not normalized_model_name.startswith("facebook/"):
            raise RuntimeError(f"DINOv3のリポジトリが見つからない: {repo_dir}")
        if weights_path is not None and str(weights_path) and Path(weights_path).exists() is False and normalized_model_name.startswith("facebook/") is False:
            raise RuntimeError(f"DINOv3の重みが見つからない: {weights_path}")

        self._torch = torch
        self.model_name = model_name
        self.model_id = normalized_model_name
        self.crop_size = crop_size
        self.weights_sha256 = sha256_of_file(weights_path) if weights_path is not None and Path(weights_path).is_file() else ""
        self._mean_std = PRETRAIN_NORMALIZATION[pretrain_dataset]
        self.device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

        try:
            model = AutoModel.from_pretrained(self.model_id)
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
        cropped = prepare_crop(sample, self.crop_size)
        pixels = np.asarray(cropped.rgb, dtype=np.float32) / MASK_FOREGROUND
        tensor = self._torch.from_numpy(pixels).permute(2, 0, 1)
        mean = self._torch.tensor(self._mean_std[0], dtype=tensor.dtype).view(3, 1, 1)
        std = self._torch.tensor(self._mean_std[1], dtype=tensor.dtype).view(3, 1, 1)
        mask = self._torch.from_numpy(np.asarray(cropped.mask, dtype=np.float32) / MASK_FOREGROUND)
        batch = (((tensor - mean) / std) * mask.unsqueeze(0)).unsqueeze(0).to(self.device)

        with self._torch.inference_mode():
            layers = self._intermediate_layers(batch)

        grid = self.crop_size // self.patch_size
        block_features: list[np.ndarray] = []
        for token_map in layers:
            if token_map.ndim == 3:
                token_map = token_map[0]
            if token_map.shape[0] != grid * grid:
                raise DataContractError(f"patch tokenは{grid * grid}個必要だが、{token_map.shape[0]}個である")
            block_features.append(patch_tokens_to_hwc(token_map.float().cpu().numpy(), grid, grid))

        features = block_features[0]
        self._save_array(destination, features)
        return FeatureArtifact(
            path=destination,
            shape=(int(features.shape[0]), int(features.shape[1]), int(features.shape[2])),
            dtype=str(features.dtype),
            model_name=self.model_name,
            weights_sha256=self.weights_sha256,
        )

    def _intermediate_layers(self, images: object) -> list[np.ndarray]:
        model = self.model
        if hasattr(model, "get_intermediate_layers"):
            layer_indices = self._layer_indices()
            outputs = model.get_intermediate_layers(images, n=layer_indices, reshape=True, norm=True)
            return [np.asarray(item[0].float().cpu(), dtype=np.float32) for item in outputs]

        if hasattr(model, "forward"):
            outputs = model(images, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.get("hidden_states")
            if hidden_states is not None:
                selected = []
                for index in self._layer_indices():
                    token_map = hidden_states[index - 1]
                    selected.append(np.asarray(token_map[0].float().cpu(), dtype=np.float32))
                return selected
        raise DataContractError("対応するTransformersの中間層APIが見つからない")

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
