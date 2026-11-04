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

from dinov3_pipeline.domain.models import DataContractError
from dinov3_pipeline.infrastructure.pca_visualizer import save_pca_visualization

IMAGE_SIZE = 32
RGB_COLOR = (100, 120, 140)


def _write_inputs(
    tmp_path: Path, mask_value: int, grid: int
) -> tuple[Path, Path, Path]:
    """特徴量・RGB・maskを書き出す。

    Args:
        tmp_path: 保存先。
        mask_value: maskの塗りつぶし値。
        grid: patch格子の辺長。

    Returns:
        特徴量・RGB・maskのパス。
    """
    feature_path = tmp_path / "feature.npy"
    rgb_path = tmp_path / "rgb.png"
    mask_path = tmp_path / "mask.png"
    features = np.arange(grid * grid * 5, dtype=np.float32).reshape(grid, grid, 5)
    np.save(feature_path, features, allow_pickle=False)
    Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), RGB_COLOR).save(rgb_path)
    Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), mask_value).save(mask_path)
    return feature_path, rgb_path, mask_path


def test_save_pca_visualization_foreground_exists_writes_side_by_side_png(
    tmp_path: Path,
) -> None:
    """crop RGBとPCA画像を横に並べたPNGを保存する。"""
    # 準備
    feature_path, rgb_path, mask_path = _write_inputs(tmp_path, mask_value=255, grid=4)
    destination = tmp_path / "debug" / "pca.png"

    # 実行
    save_pca_visualization(feature_path, rgb_path, mask_path, destination)

    # 検証
    with Image.open(destination) as output:
        assert output.size == (IMAGE_SIZE * 2, IMAGE_SIZE)


def test_save_pca_visualization_foreground_exists_keeps_rgb_on_left(
    tmp_path: Path,
) -> None:
    """左半分にcrop RGBをそのまま置く。"""
    # 準備
    feature_path, rgb_path, mask_path = _write_inputs(tmp_path, mask_value=255, grid=4)
    destination = tmp_path / "pca.png"

    # 実行
    save_pca_visualization(feature_path, rgb_path, mask_path, destination)

    # 検証
    with Image.open(destination) as output:
        assert output.getpixel((0, 0)) == RGB_COLOR


def test_save_pca_visualization_foreground_too_small_raises_contract_error(
    tmp_path: Path,
) -> None:
    """前景patchが3個未満ならPCAを実行せず拒否する。"""
    # 準備
    feature_path, rgb_path, mask_path = _write_inputs(tmp_path, mask_value=0, grid=2)

    # 実行・検証
    with pytest.raises(DataContractError, match="前景patch"):
        save_pca_visualization(feature_path, rgb_path, mask_path, tmp_path / "pca.png")
