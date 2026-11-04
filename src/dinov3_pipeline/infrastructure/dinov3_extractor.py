"""torch.hub のローカル重みでDINOv3 ViTを実行し、patch特徴量をHWC形式で保存する。

責務:
    固定cropのRGBからDINOv3のpatch tokenを抽出し、float32 HWCのNPYとして保存する。
    承認状態・原本hashの検証や、CNN向けの出力は担わない。

処理の流れ:
    1. 設定のリポジトリ・モデル名・重みから、torch.hub でバックボーンを読み込む。
    2. prepare_crop で切り出したRGBを、重みに合う統計で正規化し、maskで背景を厳密に0にする。
    3. forward_features のpatch tokenを (H, W, C) に並べ直して保存する。

関連ファイル:
    infrastructure/preprocessing.py、infrastructure/file_hash.py、domain/ports.py（FeatureExtractor）、
    usecase/delivery.py、constants.py（対応モデル・正規化統計）。

設計意図:
    Hugging Face認証やTransformersに依存せず、Metaが配布した重みをPyTorchだけで読む。
    モデル種別・重み・事前学習データは設定で切り替え、出力には重みのSHA-256を残して再現性を担保する。

Note:
    facebookresearch/dinov3 のclone（DINOv3 License）と、Meta配布ページから取得した重みが必要。
    GPU（CUDA）があれば自動で使う。depthはRGB専用のbackboneには入力しない。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from dinov3_pipeline.constants import (
    HUB_SOURCE,
    MASK_FOREGROUND,
    PATCH_TOKENS_KEY,
    PRETRAIN_NORMALIZATION,
)
from dinov3_pipeline.domain.models import (
    DataContractError,
    FeatureArtifact,
    ReadySample,
)
from dinov3_pipeline.infrastructure.file_hash import sha256_of_file
from dinov3_pipeline.infrastructure.preprocessing import prepare_crop


def patch_tokens_to_hwc(
    patch_tokens: np.ndarray, grid_height: int, grid_width: int
) -> np.ndarray:
    """行優先のViT patch tokenを、検証済みのfloat32 HWC配列に変換する。

    Args:
        patch_tokens: ``(grid_height * grid_width, channels)`` 形状のtoken行列。
        grid_height: 縦方向のpatch数。
        grid_width: 横方向のpatch数。

    Returns:
        ``(grid_height, grid_width, channels)`` 形状の空間特徴量。

    Raises:
        DataContractError: token数またはchannel数が不正な場合。
    """
    if grid_height <= 0 or grid_width <= 0 or patch_tokens.ndim != 2:
        raise DataContractError(
            "patch tokenには空でない2次元行列と正のgrid寸法が必要である"
        )
    # 数が合わないまま並べ直すと空間位置がずれるため、reshape前に拒否する
    if patch_tokens.shape[0] != grid_height * grid_width or patch_tokens.shape[1] <= 0:
        raise DataContractError("patch token数が指定された空間gridと一致しない")
    return np.asarray(
        patch_tokens.reshape(grid_height, grid_width, -1), dtype=np.float32
    )


class Dinov3FeatureExtractor:
    """固定cropをリサイズせずに処理し、DINOv3のpatch tokenを抽出する。

    Args:
        repo_dir: facebookresearch/dinov3 をcloneしたディレクトリ。
        model_name: hubconf.py のエントリ名（例: dinov3_vitb16）。
        weights_path: Meta配布のチェックポイント（.pth）。
        pretrain_dataset: 重みの事前学習データ（正規化統計の選択に使う）。
        crop_size: 正方形cropの辺長。patch sizeで割り切れる必要がある。
        device_name: PyTorchのdevice名。Noneなら自動選択する。

    Raises:
        RuntimeError: torchが無い、リポジトリ・重みが無い、またはモデルを読み込めない場合。
        ValueError: crop_sizeがpatch sizeで割り切れない場合。
    """

    def __init__(
        self,
        repo_dir: Path,
        model_name: str,
        weights_path: Path,
        pretrain_dataset: str,
        crop_size: int,
        device_name: str | None,
    ) -> None:
        # 重い依存は抽出器を作るときだけ読み込み、他の処理とテストを軽く保つ
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("DINOv3の推論にはtorchが必要である") from error
        if not repo_dir.is_dir():
            raise RuntimeError(
                f"DINOv3のリポジトリが見つからない: {repo_dir}。"
                "facebookresearch/dinov3 をcloneして repo_dir に設定する"
            )
        if not weights_path.is_file():
            raise RuntimeError(
                f"DINOv3の重みが見つからない: {weights_path}。"
                "Metaの配布ページから取得して weights_path に設定する"
            )

        self._torch = torch
        self.model_name = model_name
        self.crop_size = crop_size
        # 同じモデル名でも重みが違えば別の特徴量になるため、内容のhashで識別する
        self.weights_sha256 = sha256_of_file(weights_path)
        self._mean_std = PRETRAIN_NORMALIZATION[pretrain_dataset]
        self.device = torch.device(
            device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        try:
            # torch.hub.load は型注釈を持たないため、ここだけ型検査を外す
            model = torch.hub.load(  # type: ignore[no-untyped-call]
                str(repo_dir), model_name, source=HUB_SOURCE, weights=str(weights_path)
            )
        except Exception as error:
            # hub側は例外の種類が多岐にわたるため、原因を保ったまま1種類に揃える
            raise RuntimeError(
                f"DINOv3を読み込めない: {model_name} / {weights_path}（{error}）"
            ) from error
        self.model = model.to(self.device)
        self.model.eval()
        self.patch_size = int(self.model.patch_size)
        if crop_size <= 0 or crop_size % self.patch_size:
            raise ValueError(
                f"crop_sizeは正の値で、patch size {self.patch_size}で割り切れる必要がある"
            )

    def extract(self, sample: ReadySample, destination: Path) -> FeatureArtifact:
        """float32 HWC patch mapを保存し、生成元の情報を返す。

        Args:
            sample: 承認済み物体。
            destination: 特徴量ファイルの保存先。

        Returns:
            保存した特徴量のshape・dtypeとモデル識別情報。

        Raises:
            DataContractError: 入力が不正、またはモデル出力のtoken数が想定と違う場合。
        """
        cropped = prepare_crop(sample, self.crop_size)
        pixels = np.asarray(cropped.rgb, dtype=np.float32) / MASK_FOREGROUND
        tensor = self._torch.from_numpy(pixels).permute(2, 0, 1)
        mean = self._torch.tensor(self._mean_std[0], dtype=tensor.dtype).view(3, 1, 1)
        std = self._torch.tensor(self._mean_std[1], dtype=tensor.dtype).view(3, 1, 1)
        mask = self._torch.from_numpy(
            np.asarray(cropped.mask, dtype=np.float32) / MASK_FOREGROUND
        )
        # 正規化の後にmaskを掛け、背景が丸め誤差なく0になった状態でbackboneへ渡す
        batch = (
            (((tensor - mean) / std) * mask.unsqueeze(0)).unsqueeze(0).to(self.device)
        )

        with self._torch.inference_mode():
            output = self.model.forward_features(batch)
        if PATCH_TOKENS_KEY not in output:
            raise DataContractError(f"DINOv3の出力に{PATCH_TOKENS_KEY}が含まれていない")

        grid = self.crop_size // self.patch_size
        # CLSとregister tokenを除いたpatch tokenだけを空間特徴量として使う
        patch_tokens = output[PATCH_TOKENS_KEY][0]
        if patch_tokens.shape[0] != grid * grid:
            raise DataContractError(
                f"patch tokenは{grid * grid}個必要だが、{patch_tokens.shape[0]}個である"
            )
        features = patch_tokens_to_hwc(patch_tokens.float().cpu().numpy(), grid, grid)
        self._save_array(destination, features)
        return FeatureArtifact(
            path=destination,
            shape=(
                int(features.shape[0]),
                int(features.shape[1]),
                int(features.shape[2]),
            ),
            dtype=str(features.dtype),
            model_name=self.model_name,
            weights_sha256=self.weights_sha256,
        )

    @staticmethod
    def _save_array(destination: Path, features: np.ndarray) -> None:
        """pickle無効のNPYを、一時ファイル経由で保存先へ反映する。

        Args:
            destination: 保存先。
            features: 保存する特徴量。
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        # 書き込み途中のファイルを他の処理が読まないよう、完成後に置き換える
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=destination.suffix, delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.save(stream, features, allow_pickle=False)
        temporary.replace(destination)
