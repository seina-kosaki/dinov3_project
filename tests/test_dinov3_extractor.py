"""DINOv3抽出器（torch.hub のローカル重み方式）を、ダミーモデルで確認する。

責務:
    Dinov3FeatureExtractor が、設定どおりにモデルを読み込み、HWC特徴量を保存し、
    背景を0にして正規化することを確認する。

処理の流れ:
    torch.hub.load をダミーのバックボーンへ差し替え、抽出結果と入力テンソルを検証する。

関連ファイル:
    src/dinov3_pipeline/infrastructure/dinov3_extractor.py、tests/conftest.py。

設計意図:
    実重み（数百MB以上）や公式リポジトリを使わず、モデルの入出力契約と設定の効き方を確認する。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from conftest import RGB_COLOR, SampleFactory

from dinov3_pipeline.constants import PRETRAIN_NORMALIZATION
from dinov3_pipeline.domain.models import FeatureArtifact
from dinov3_pipeline.infrastructure.dinov3_extractor import Dinov3FeatureExtractor
from dinov3_pipeline.infrastructure.file_hash import sha256_of_file

CROP_SIZE = 32
PATCH_SIZE = 16
HIDDEN_SIZE = 8
MODEL_NAME = "dinov3_vitb16"
GRID = CROP_SIZE // PATCH_SIZE


class FakeBackbone(torch.nn.Module):
    """公式バックボーンと同じ forward_features の形を返すダミーモデルである。"""

    patch_size = PATCH_SIZE

    def __init__(self) -> None:
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """入力を記録し、patch tokenを持つ辞書を返す。

        Args:
            x: 入力バッチ。

        Returns:
            ``x_norm_patchtokens`` を持つ辞書。
        """
        self.last_input = x
        batch, _, height, width = x.shape
        count = (height // PATCH_SIZE) * (width // PATCH_SIZE)
        return {"x_norm_patchtokens": torch.ones(batch, count, HIDDEN_SIZE)}


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path]:
    """公式リポジトリ相当のディレクトリと、重みファイルを作る。

    Args:
        tmp_path: 一時ディレクトリ。

    Returns:
        リポジトリのディレクトリと、重みファイルのパス。
    """
    repo_dir = tmp_path / "dinov3"
    repo_dir.mkdir()
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"weights")
    return repo_dir, weights


@contextmanager
def _hub(model: torch.nn.Module) -> Iterator[MagicMock]:
    """torch.hub.load を、指定モデルを返すダミーへ差し替える。

    Args:
        model: 返すダミーモデル。

    Yields:
        差し替えた load のモック。
    """
    with patch("torch.hub.load", return_value=model) as load:
        yield load


def _extractor(
    paths: tuple[Path, Path], dataset: str = "lvd1689m", crop_size: int = CROP_SIZE
) -> Dinov3FeatureExtractor:
    """ダミーモデルで抽出器を作る。

    Args:
        paths: リポジトリと重み。
        dataset: 事前学習データ名。
        crop_size: crop辺長。

    Returns:
        抽出器。
    """
    return Dinov3FeatureExtractor(
        paths[0], MODEL_NAME, paths[1], dataset, crop_size, "cpu"
    )


def test_init_valid_config_loads_model_from_local_repo_and_weights(
    paths: tuple[Path, Path],
) -> None:
    """設定のリポジトリ・モデル名・重みで torch.hub から読み込む。"""
    # 準備
    repo_dir, weights = paths

    # 実行
    with _hub(FakeBackbone()) as load:
        _extractor(paths)

    # 検証
    load.assert_called_once_with(
        str(repo_dir), MODEL_NAME, source="local", weights=str(weights)
    )


def test_init_missing_repo_dir_raises_runtime_error(
    paths: tuple[Path, Path], tmp_path: Path
) -> None:
    """公式リポジトリが無い場合は、読み込み前に拒否する。"""
    # 準備
    missing = (tmp_path / "missing", paths[1])

    # 実行・検証
    with pytest.raises(RuntimeError, match="リポジトリが見つからない"):
        _extractor(missing)


def test_init_missing_weights_raises_runtime_error(
    paths: tuple[Path, Path], tmp_path: Path
) -> None:
    """重みが無い場合は、読み込み前に拒否する。"""
    # 準備
    missing = (paths[0], tmp_path / "missing.pth")

    # 実行・検証
    with pytest.raises(RuntimeError, match="重みが見つからない"):
        _extractor(missing)


def test_init_hub_failure_raises_runtime_error(paths: tuple[Path, Path]) -> None:
    """hub側の読み込み失敗を RuntimeError に揃える。"""
    # 準備・実行・検証
    with (
        patch("torch.hub.load", side_effect=ValueError("bad weights")),
        pytest.raises(RuntimeError, match="読み込めない"),
    ):
        _extractor(paths)


def test_init_crop_size_not_multiple_of_patch_raises_value_error(
    paths: tuple[Path, Path],
) -> None:
    """patch sizeで割り切れないcropサイズを拒否する。"""
    # 準備・実行・検証
    with _hub(FakeBackbone()), pytest.raises(ValueError, match="割り切れる"):
        _extractor(paths, crop_size=PATCH_SIZE + 1)


@pytest.fixture
def extracted(
    paths: tuple[Path, Path], make_sample: SampleFactory, tmp_path: Path
) -> tuple[FeatureArtifact, Path, FakeBackbone]:
    """物体の周囲に背景があるsampleを抽出し、結果とダミーモデルを返す。

    Args:
        paths: リポジトリと重み。
        make_sample: 承認済み物体の生成関数。
        tmp_path: 一時ディレクトリ。

    Returns:
        抽出結果、特徴量ファイル、入力を記録したダミーモデル。
    """
    backbone = FakeBackbone()
    sample = make_sample(size=64, mask_box=(24, 24, 40, 40))
    destination = tmp_path / "features" / "sample.npy"
    with _hub(backbone):
        artifact = _extractor(paths).extract(sample, destination)
    return artifact, destination, backbone


def test_extract_valid_sample_returns_patch_grid_shape(
    extracted: tuple[FeatureArtifact, Path, FakeBackbone],
) -> None:
    """crop辺長/patch sizeのgridと、特徴次元のshapeを返す。"""
    # 準備・実行
    artifact, _, _ = extracted

    # 検証
    assert artifact.shape == (GRID, GRID, HIDDEN_SIZE)


def test_extract_valid_sample_saves_float32_npy(
    extracted: tuple[FeatureArtifact, Path, FakeBackbone],
) -> None:
    """特徴量を、有限値のfloat32 NPYとして保存する。"""
    # 準備・実行
    _, destination, _ = extracted

    # 検証
    features = np.load(destination, allow_pickle=False)
    assert features.dtype == np.float32 and np.isfinite(features).all()


def test_extract_valid_sample_records_model_name_and_weights_hash(
    extracted: tuple[FeatureArtifact, Path, FakeBackbone], paths: tuple[Path, Path]
) -> None:
    """生成元として、モデル名と重みのSHA-256を返す。"""
    # 準備・実行
    artifact, _, _ = extracted

    # 検証
    assert (artifact.model_name, artifact.weights_sha256) == (
        MODEL_NAME,
        sha256_of_file(paths[1]),
    )


def test_extract_valid_sample_inputs_exact_zero_for_background(
    extracted: tuple[FeatureArtifact, Path, FakeBackbone],
) -> None:
    """正規化後の背景を、厳密に0にしてモデルへ渡す。"""
    # 準備・実行
    _, _, backbone = extracted

    # 検証
    assert backbone.last_input is not None
    assert torch.count_nonzero(backbone.last_input[0, :, 0, 0]) == 0


@pytest.mark.parametrize("dataset", sorted(PRETRAIN_NORMALIZATION))
def test_extract_pretrain_dataset_normalizes_with_matching_statistics(
    paths: tuple[Path, Path],
    make_sample: SampleFactory,
    tmp_path: Path,
    dataset: str,
) -> None:
    """事前学習データに対応する平均・標準偏差で正規化する。"""
    # 準備
    backbone = FakeBackbone()
    sample = make_sample(size=64, mask_box=(24, 24, 40, 40))
    mean, std = PRETRAIN_NORMALIZATION[dataset]

    # 実行
    with _hub(backbone):
        _extractor(paths, dataset).extract(sample, tmp_path / "feature.npy")

    # 検証
    assert backbone.last_input is not None
    expected = [(value / 255 - m) / s for value, m, s in zip(RGB_COLOR, mean, std)]
    center = backbone.last_input[0, :, CROP_SIZE // 2, CROP_SIZE // 2]
    np.testing.assert_allclose(center.numpy(), expected, rtol=1e-5)
