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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from conftest import RGB_COLOR, SampleFactory

from dinov3_pipeline.common.constants import PRETRAIN_NORMALIZATION
from dinov3_pipeline.common.file_hash import sha256_of_file
from dinov3_pipeline.common.models import FeatureArtifact
from dinov3_pipeline.dinov3.extractor import Dinov3FeatureExtractor

CROP_SIZE = 32
PATCH_SIZE = 16
HIDDEN_SIZE = 8
MODEL_NAME = "dinov3_vitb16"
GRID = CROP_SIZE // PATCH_SIZE


class FakeBackbone(torch.nn.Module):
    """Transformers DINOv3モデルのhidden_states形式を返すダミーモデルである。"""

    patch_size = PATCH_SIZE

    def __init__(self, token_count: int | None = None) -> None:
        super().__init__()
        self.last_input: torch.Tensor | None = None
        self.layer_outputs: tuple[torch.Tensor, ...] = ()
        self.config = SimpleNamespace(
            patch_size=PATCH_SIZE,
            num_hidden_layers=12,
            num_register_tokens=4,
        )
        self.embeddings = torch.nn.Module()
        self.embeddings.register_parameter("cls_token", torch.nn.Parameter(torch.zeros(1, 1, HIDDEN_SIZE)))
        self.embeddings.register_parameter("register_tokens", torch.nn.Parameter(torch.zeros(1, 4, HIDDEN_SIZE)))
        self.norm = torch.nn.LayerNorm(HIDDEN_SIZE)
        with torch.no_grad():
            self.norm.weight.copy_(torch.arange(1, HIDDEN_SIZE + 1, dtype=torch.float32))
            self.norm.bias.copy_(torch.arange(HIDDEN_SIZE, dtype=torch.float32) / 10)
        self.token_count = token_count

    def forward(self, x: torch.Tensor, **kwargs: object) -> dict[str, tuple[torch.Tensor, ...]]:
        self.last_input = x
        batch = x.shape[0]
        count = self.token_count or (GRID * GRID + 5)
        channel_values = torch.arange(HIDDEN_SIZE, dtype=torch.float32).view(1, 1, HIDDEN_SIZE)
        patch_values = torch.arange(1, count - 4, dtype=torch.float32).view(1, -1, 1)
        outputs = []
        for layer in range(self.config.num_hidden_layers + 1):
            tokens = torch.zeros(batch, count, HIDDEN_SIZE)
            tokens[:, :5] = 900 + layer
            if count > 5:
                tokens[:, 5:] = layer * 10 + patch_values + channel_values / 10
            outputs.append(tokens)
        self.layer_outputs = tuple(outputs)
        return {
            "hidden_states": self.layer_outputs,
            "last_hidden_state": self.norm(self.layer_outputs[-1]),
        }


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
    """Transformersのモデルロードを、指定モデルを返すダミーへ差し替える。

    Args:
        model: 返すダミーモデル。

    Yields:
        差し替えた load のモック。
    """
    with patch("transformers.AutoModel.from_pretrained", return_value=model) as load:
        yield load


def _extractor(paths: tuple[Path, Path], dataset: str = "lvd1689m", crop_size: int = CROP_SIZE) -> Dinov3FeatureExtractor:
    """ダミーモデルで抽出器を作る。

    Args:
        paths: リポジトリと重み。
        dataset: 事前学習データ名。
        crop_size: crop辺長。

    Returns:
        抽出器。
    """
    return Dinov3FeatureExtractor(paths[0], MODEL_NAME, paths[1], dataset, crop_size, "cpu")


def test_init_passes_hf_token_to_from_pretrained(monkeypatch: pytest.MonkeyPatch) -> None:
    """環境変数で解決したHF tokenを、モデル読み込みにそのまま受け渡す。"""

    class DummyModel:
        config = type("Config", (), {"patch_size": 16, "num_hidden_layers": 12})()

        def to(self, device: str) -> DummyModel:
            return self

        def eval(self) -> None:
            return None

    monkeypatch.setenv("HF_TOKEN", "hf_secret_token")
    with patch("transformers.AutoModel.from_pretrained", return_value=DummyModel()) as mocked:
        Dinov3FeatureExtractor(None, MODEL_NAME, None, "lvd1689m", CROP_SIZE, "cpu")

    mocked.assert_called_once_with("facebook/dinov3-vitb16-pretrain-lvd1689m", token="hf_secret_token")


def test_init_valid_config_loads_model_from_local_repo_and_weights(
    paths: tuple[Path, Path],
) -> None:
    """旧設定名をTransformersの正規モデルIDへ変換して読み込む。"""
    with _hub(FakeBackbone()) as load:
        _extractor(paths)

    load.assert_called_once_with("facebook/dinov3-vitb16-pretrain-lvd1689m")


def test_init_missing_repo_dir_raises_runtime_error(paths: tuple[Path, Path], tmp_path: Path) -> None:
    """公式リポジトリが無い場合は、読み込み前に拒否する。"""
    # 準備
    missing = (tmp_path / "missing", paths[1])

    # 実行・検証
    with pytest.raises(RuntimeError, match="リポジトリが見つからない"):
        Dinov3FeatureExtractor(missing[0], "local-model", missing[1], "lvd1689m", CROP_SIZE, "cpu")


def test_init_missing_weights_raises_runtime_error(paths: tuple[Path, Path], tmp_path: Path) -> None:
    """重みが無い場合は、読み込み前に拒否する。"""
    # 準備
    missing = (paths[0], tmp_path / "missing.pth")

    # 実行・検証
    with pytest.raises(RuntimeError, match="重みが見つからない"):
        Dinov3FeatureExtractor(missing[0], "local-model", missing[1], "lvd1689m", CROP_SIZE, "cpu")


def test_init_transformers_failure_raises_runtime_error(paths: tuple[Path, Path]) -> None:
    """Transformers側の読み込み失敗を RuntimeError に揃える。"""
    with (
        patch("transformers.AutoModel.from_pretrained", side_effect=ValueError("bad weights")),
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
def extracted(paths: tuple[Path, Path], make_sample: SampleFactory, tmp_path: Path) -> tuple[FeatureArtifact, Path, FakeBackbone]:
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
    assert len(extracted[0].block_paths) == 4
    assert all(path.is_file() for path in extracted[0].block_paths)


def test_extract_valid_sample_records_model_name_and_weights_hash(extracted: tuple[FeatureArtifact, Path, FakeBackbone], paths: tuple[Path, Path]) -> None:
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


def test_extract_removes_model_reported_special_tokens_before_saving(
    paths: tuple[Path, Path],
    make_sample: SampleFactory,
    tmp_path: Path,
) -> None:
    """CLSとregister tokenを除き、patchだけを行優先のHWC gridに保存する。"""
    backbone = FakeBackbone()
    sample = make_sample(size=64, mask_box=(24, 24, 40, 40))
    destination = tmp_path / "feature.npy"

    with _hub(backbone):
        _extractor(paths).extract(sample, destination)

    features = np.load(destination, allow_pickle=False)
    assert features.shape == (GRID, GRID, HIDDEN_SIZE)
    with torch.inference_mode():
        expected = backbone.norm(backbone.layer_outputs[-1])[:, 5:, :].reshape(
            GRID, GRID, HIDDEN_SIZE
        ).numpy()
    np.testing.assert_allclose(features, expected)


def test_extract_saves_quartile_block_outputs_in_order_before_final_layer_norm(
    extracted: tuple[FeatureArtifact, Path, FakeBackbone],
) -> None:
    artifact, _, backbone = extracted
    layers = (3, 6, 9, 12)

    assert [path.parent.name for path in artifact.block_paths] == [
        "feature_block_1",
        "feature_block_2",
        "feature_block_3",
        "feature_block_4",
    ]
    for path, layer in zip(artifact.block_paths, layers):
        actual = np.load(path, allow_pickle=False)
        expected = backbone.layer_outputs[layer][:, 5:, :].reshape(
            GRID, GRID, HIDDEN_SIZE
        ).numpy()
        np.testing.assert_array_equal(actual, expected)


def test_extract_final_feature_is_final_block_output_after_final_layer_norm(
    extracted: tuple[FeatureArtifact, Path, FakeBackbone],
) -> None:
    artifact, destination, backbone = extracted
    final_feature = np.load(destination, allow_pickle=False)
    final_block = np.load(artifact.block_paths[-1], allow_pickle=False)

    assert not np.array_equal(final_feature, final_block)
    with torch.inference_mode():
        expected = backbone.norm(backbone.layer_outputs[12])[:, 5:, :].reshape(
            GRID, GRID, HIDDEN_SIZE
        ).numpy()
    np.testing.assert_allclose(final_feature, expected)


def test_extract_rejects_output_when_special_token_boundary_is_unknown(
    paths: tuple[Path, Path],
    make_sample: SampleFactory,
    tmp_path: Path,
) -> None:
    """モデル情報からspecial tokenの境界を得られない出力を拒否する。"""
    backbone = FakeBackbone()
    del backbone.embeddings
    sample = make_sample(size=64, mask_box=(24, 24, 40, 40))

    with _hub(backbone), pytest.raises(ValueError, match="境界をモデル情報から識別できない"):
        _extractor(paths).extract(sample, tmp_path / "feature.npy")


def test_extract_rejects_invalid_special_and_patch_token_count(
    paths: tuple[Path, Path],
    make_sample: SampleFactory,
    tmp_path: Path,
) -> None:
    """special token数を除いてもgrid数に一致しない出力を拒否する。"""
    backbone = FakeBackbone(token_count=10)
    sample = make_sample(size=64, mask_box=(24, 24, 40, 40))

    with _hub(backbone), pytest.raises(ValueError, match="patch tokenは4個必要"):
        _extractor(paths).extract(sample, tmp_path / "feature.npy")


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
