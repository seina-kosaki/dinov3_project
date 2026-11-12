"""PCA-RGBのdebug画像が、HWC特徴量から正しく生成されることを検証する。

責務:
    save_pca_visualization の出力形式と、前景patch不足時の拒否を確認する。

処理の流れ:
    小さなNPY・RGB・maskを作り、可視化を呼んで出力PNGを検証する。

関連ファイル:
    src/dinov3_pipeline/infrastructure/pca_visualizer.py。

設計意図:
    モデルを使わず、PCAの入出力契約だけを高速に確認する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dinov3_pipeline.common.models import DataContractError
from dinov3_pipeline.saving.pca_visualizer import save_pca_visualization

IMAGE_SIZE = 32
RGB_COLOR = (100, 120, 140)


def _write_inputs(
    tmp_path: Path, mask_value: int, grid: int
) -> tuple[tuple[Path, ...], Path, Path]:
    """特徴量・RGB・maskを書き出す。

    Args:
        tmp_path: 保存先。
        mask_value: maskの塗りつぶし値。
        grid: patch格子の辺長。

    Returns:
        特徴量・RGB・maskのパス。
    """
    feature_paths = tuple(tmp_path / f"feature_{index}.npy" for index in range(5))
    rgb_path = tmp_path / "rgb.png"
    mask_path = tmp_path / "mask.png"
    for index, feature_path in enumerate(feature_paths):
        features = np.arange(grid * grid * 5, dtype=np.float32).reshape(grid, grid, 5) + index
        np.save(feature_path, features, allow_pickle=False)
    Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), RGB_COLOR).save(rgb_path)
    Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), mask_value).save(mask_path)
    return feature_paths, rgb_path, mask_path


def test_save_pca_visualization_foreground_exists_writes_six_panels_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RGBと5特徴量の可視化を指定順で3列×2行に配置する。"""
    feature_paths, rgb_path, mask_path = _write_inputs(tmp_path, mask_value=255, grid=4)
    destination = tmp_path / "debug" / "pca.png"
    colors = iter([(20, 0, 0), (40, 0, 0), (60, 0, 0), (80, 0, 0), (100, 0, 0)])
    monkeypatch.setattr(
        "dinov3_pipeline.saving.pca_visualizer._pca_colors",
        lambda vectors: np.tile(next(colors), (len(vectors), 1)).astype(np.uint8),
    )
    save_pca_visualization(feature_paths, rgb_path, mask_path, destination)
    with Image.open(destination) as output:
        assert output.size == (IMAGE_SIZE * 3, IMAGE_SIZE * 2)
        assert output.getpixel((0, 0)) == RGB_COLOR
        assert [
            output.getpixel(((index % 3) * IMAGE_SIZE, (index // 3) * IMAGE_SIZE))
            for index in range(1, 6)
        ] == [(20, 0, 0), (40, 0, 0), (60, 0, 0), (80, 0, 0), (100, 0, 0)]


def test_save_pca_visualization_foreground_too_small_raises_contract_error(
    tmp_path: Path,
) -> None:
    """前景patchが3個未満ならPCAを実行せず拒否する。"""
    # 準備
    feature_paths, rgb_path, mask_path = _write_inputs(tmp_path, mask_value=0, grid=2)

    # 実行・検証
    with pytest.raises(DataContractError, match="前景patch"):
        save_pca_visualization(feature_paths, rgb_path, mask_path, tmp_path / "pca.png")


def test_save_pca_visualization_without_rgb_mask_uses_all_patches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RGB背景除去無効時は背景maskに関係なく全patchでPCAする。"""
    feature_paths, rgb_path, mask_path = _write_inputs(tmp_path, mask_value=0, grid=2)
    observed_counts: list[int] = []
    monkeypatch.setattr(
        "dinov3_pipeline.saving.pca_visualizer._pca_colors",
        lambda vectors: observed_counts.append(len(vectors))
        or np.zeros((len(vectors), 3), dtype=np.uint8),
    )

    destination = tmp_path / "pca_unmasked.png"
    save_pca_visualization(
        feature_paths,
        rgb_path,
        mask_path,
        destination,
        mask_rgb_background=False,
    )

    assert observed_counts == [4] * 5
    with Image.open(destination) as output:
        assert output.getpixel((0, 0)) == RGB_COLOR


def test_save_pca_visualization_failure_does_not_save_partial_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """どれかの特徴量でPCAが失敗した場合、部分画像を保存しない。"""
    feature_paths, rgb_path, mask_path = _write_inputs(tmp_path, mask_value=255, grid=4)
    destination = tmp_path / "pca.png"
    calls = 0

    def fail_on_second_feature(vectors: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise DataContractError("PCA failed")
        return np.zeros((len(vectors), 3), dtype=np.uint8)

    monkeypatch.setattr("dinov3_pipeline.saving.pca_visualizer._pca_colors", fail_on_second_feature)
    with pytest.raises(DataContractError, match="PCA failed"):
        save_pca_visualization(feature_paths, rgb_path, mask_path, destination)
    assert not destination.exists()
