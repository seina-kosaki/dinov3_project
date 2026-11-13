"""DINOv3抽出器（Transformersモデル取得）を、ダミーモデルで確認する。

責務:
    Dinov3FeatureExtractor が、設定どおりにモデルを読み込み、HWC特徴量を保存し、
    背景を0にして正規化することを確認する。

処理の流れ:
    Transformersのモデルロードをダミーのバックボーンへ差し替え、抽出結果と入力テンソルを検証する。

関連ファイル:
    src/dinov3_pipeline/dinov3/extractor.py、tests/conftest.py。

設計意図:
    実モデルや外部サービスを使わず、モデルの入出力契約と設定の効き方を確認する。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from conftest import RGB_COLOR
from PIL import Image

from dinov3_pipeline.common.constants import PRETRAIN_NORMALIZATION
from dinov3_pipeline.dinov3.extractor import Dinov3FeatureExtractor

CROP_SIZE = 32
PATCH_SIZE = 16
HIDDEN_SIZE = 8
MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
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


def _extractor(
    dataset: str = "lvd1689m",
) -> Dinov3FeatureExtractor:
    """ダミーモデルで抽出器を作る。

    Args:
        dataset: 事前学習データ名。
        crop_size: crop辺長。

    Returns:
        抽出器。
    """
    return Dinov3FeatureExtractor(MODEL_NAME, dataset, "cpu")


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
        Dinov3FeatureExtractor(MODEL_NAME, "lvd1689m", "cpu")

    mocked.assert_called_once_with(MODEL_NAME, token="hf_secret_token")


def test_init_loads_model_from_official_model_id() -> None:
    """設定された公式モデルIDをTransformersで読み込む。"""
    with _hub(FakeBackbone()) as load:
        _extractor()

    load.assert_called_once_with(MODEL_NAME)


@pytest.mark.parametrize("model_name", ["dinov3_vitb16", "unknown/model"])
def test_init_rejects_non_official_model_id(model_name: str) -> None:
    with pytest.raises(ValueError, match="公式Hugging Face ID"):
        Dinov3FeatureExtractor(model_name, "lvd1689m", "cpu")


def test_init_transformers_failure_raises_runtime_error() -> None:
    """Transformers側の読み込み失敗を RuntimeError に揃える。"""
    with (
        patch("transformers.AutoModel.from_pretrained", side_effect=ValueError("bad weights")),
        pytest.raises(RuntimeError, match="読み込めない"),
    ):
        _extractor()


def test_init_crop_size_not_multiple_of_patch_raises_value_error() -> None:
    """patch sizeで割り切れないcropサイズを拒否する。"""
    # 準備・実行・検証
    with _hub(FakeBackbone()):
        extractor = _extractor()
    image = Image.new("RGB", (PATCH_SIZE + 1, PATCH_SIZE), RGB_COLOR)
    with pytest.raises(ValueError, match="割り切れる"):
        extractor.infer(image)


@pytest.fixture
def extracted() -> tuple[dict[str, np.ndarray], FakeBackbone]:
    """整形済みRGBとmaskを推論し、特徴量とダミーモデルを返す。

    Args:
    Returns:
        特徴量辞書と入力を記録したダミーモデル。
    """
    backbone = FakeBackbone()
    image = Image.new("RGB", (CROP_SIZE, CROP_SIZE), RGB_COLOR)
    mask = Image.new("L", (CROP_SIZE, CROP_SIZE), 0)
    mask.paste(255, (8, 8, 24, 24))
    with _hub(backbone):
        features = _extractor().infer(image, mask)
    return features, backbone


def test_extract_valid_sample_returns_patch_grid_shape(
    extracted: tuple[dict[str, np.ndarray], FakeBackbone],
) -> None:
    """crop辺長/patch sizeのgridと、特徴次元のshapeを返す。"""
    # 準備・実行
    features, _ = extracted
    assert features["feature"].shape == (GRID, GRID, HIDDEN_SIZE)


def test_extract_valid_sample_saves_float32_npy(
    extracted: tuple[dict[str, np.ndarray], FakeBackbone],
) -> None:
    """返却された全特徴量が、有限値のfloat32 HWC配列である。"""
    # 準備・実行
    features, _ = extracted
    assert set(features) == {
        "feature",
        "feature_block_1",
        "feature_block_2",
        "feature_block_3",
        "feature_block_4",
    }
    assert all(value.dtype == np.float32 for value in features.values())
    assert all(value.shape == (GRID, GRID, HIDDEN_SIZE) for value in features.values())
    assert all(np.isfinite(value).all() for value in features.values())


def test_infer_requires_rgb_mode() -> None:
    with _hub(FakeBackbone()):
        extractor = _extractor()
    with pytest.raises(ValueError, match="RGB mode"):
        extractor.infer(Image.new("L", (CROP_SIZE, CROP_SIZE)))


def test_infer_rejects_mask_with_different_dimensions() -> None:
    with _hub(FakeBackbone()):
        extractor = _extractor()
    image = Image.new("RGB", (CROP_SIZE, CROP_SIZE), RGB_COLOR)
    mask = Image.new("L", (PATCH_SIZE, PATCH_SIZE), 255)
    with pytest.raises(ValueError, match="寸法が一致"):
        extractor.infer(image, mask)


def test_extract_valid_sample_inputs_exact_zero_for_background(
    extracted: tuple[dict[str, np.ndarray], FakeBackbone],
) -> None:
    """正規化後の背景を、厳密に0にしてモデルへ渡す。"""
    # 準備・実行
    _, backbone = extracted

    # 検証
    assert backbone.last_input is not None
    assert torch.count_nonzero(backbone.last_input[0, :, 0, 0]) == 0


def test_infer_without_mask_preserves_normalized_background() -> None:
    """mask省略時は入力画素をそのまま正規化する。"""
    backbone = FakeBackbone()
    image = Image.new("RGB", (CROP_SIZE, CROP_SIZE), RGB_COLOR)
    with _hub(backbone):
        _extractor().infer(image)

    assert backbone.last_input is not None
    assert torch.count_nonzero(backbone.last_input[0, :, 0, 0]) > 0


def test_infer_removes_model_reported_special_tokens() -> None:
    """CLSとregister tokenを除き、patchだけを行優先のHWC gridにする。"""
    backbone = FakeBackbone()
    image = Image.new("RGB", (CROP_SIZE, CROP_SIZE), RGB_COLOR)

    with _hub(backbone):
        features = _extractor().infer(image)

    final_feature = features["feature"]
    assert final_feature.shape == (GRID, GRID, HIDDEN_SIZE)
    with torch.inference_mode():
        expected = backbone.norm(backbone.layer_outputs[-1])[:, 5:, :].reshape(
            GRID, GRID, HIDDEN_SIZE
        ).numpy()
    np.testing.assert_allclose(final_feature, expected)


def test_extract_saves_quartile_block_outputs_in_order_before_final_layer_norm(
    extracted: tuple[dict[str, np.ndarray], FakeBackbone],
) -> None:
    features, backbone = extracted
    layers = (3, 6, 9, 12)

    for index, layer in enumerate(layers, start=1):
        actual = features[f"feature_block_{index}"]
        expected = backbone.layer_outputs[layer][:, 5:, :].reshape(
            GRID, GRID, HIDDEN_SIZE
        ).numpy()
        np.testing.assert_array_equal(actual, expected)


def test_extract_final_feature_is_final_block_output_after_final_layer_norm(
    extracted: tuple[dict[str, np.ndarray], FakeBackbone],
) -> None:
    features, backbone = extracted
    final_feature = features["feature"]
    final_block = features["feature_block_4"]

    assert not np.array_equal(final_feature, final_block)
    with torch.inference_mode():
        expected = backbone.norm(backbone.layer_outputs[12])[:, 5:, :].reshape(
            GRID, GRID, HIDDEN_SIZE
        ).numpy()
    np.testing.assert_allclose(final_feature, expected)


def test_extract_rejects_output_when_special_token_boundary_is_unknown(
) -> None:
    """モデル情報からspecial tokenの境界を得られない出力を拒否する。"""
    backbone = FakeBackbone()
    del backbone.embeddings
    image = Image.new("RGB", (CROP_SIZE, CROP_SIZE), RGB_COLOR)

    with _hub(backbone), pytest.raises(ValueError, match="境界をモデル情報から識別できない"):
        _extractor().infer(image)


def test_extract_rejects_invalid_special_and_patch_token_count(
) -> None:
    """special token数を除いてもgrid数に一致しない出力を拒否する。"""
    backbone = FakeBackbone(token_count=10)
    image = Image.new("RGB", (CROP_SIZE, CROP_SIZE), RGB_COLOR)

    with _hub(backbone), pytest.raises(ValueError, match="patch tokenは4個必要"):
        _extractor().infer(image)


@pytest.mark.parametrize("dataset", sorted(PRETRAIN_NORMALIZATION))
def test_extract_pretrain_dataset_normalizes_with_matching_statistics(
    dataset: str,
) -> None:
    """事前学習データに対応する平均・標準偏差で正規化する。"""
    # 準備
    backbone = FakeBackbone()
    image = Image.new("RGB", (CROP_SIZE, CROP_SIZE), RGB_COLOR)
    mask = Image.new("L", (CROP_SIZE, CROP_SIZE), 255)
    mean, std = PRETRAIN_NORMALIZATION[dataset]

    # 実行
    with _hub(backbone):
        _extractor(dataset).infer(image, mask)

    # 検証
    assert backbone.last_input is not None
    expected = [(value / 255 - m) / s for value, m, s in zip(RGB_COLOR, mean, std)]
    center = backbone.last_input[0, :, CROP_SIZE // 2, CROP_SIZE // 2]
    np.testing.assert_allclose(center.numpy(), expected, rtol=1e-5)
