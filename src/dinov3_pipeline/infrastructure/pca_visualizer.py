"""保存済みHWC特徴量から、前景patchのPCA-RGB debug画像を生成する。

責務:
    特徴量NPYの前景patchを3成分PCAでRGBに変換し、crop RGBと並べたPNGを保存する。
    特徴量の生成や、CNN向けの出力は担わない。

処理の流れ:
    1. NPYを読み、maskをpatch格子に縮めて前景patchを選ぶ。
    2. 前景patchをL2正規化し、PCAの上位3成分を0〜255のRGBに変換する。
    3. crop RGBの右に、patch格子を最近傍で拡大したPCA画像を並べて保存する。

関連ファイル:
    infrastructure/delivery_exporter.py、constants.py。

設計意図:
    scikit-learnを増やさず、numpyのSVDでPCAを行う。背景は黒のままにして、前景だけの違いを見やすくする。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from dinov3_pipeline.constants import (
    COLOR_MAX,
    FEATURE_DTYPE,
    FEATURE_NDIM,
    IMAGE_FORMAT,
    MASK_FOREGROUND,
    NORM_EPSILON,
    PCA_COMPONENTS,
    PCA_MASK_THRESHOLD,
    RANGE_EPSILON,
    RGB_BACKGROUND,
)
from dinov3_pipeline.domain.models import DataContractError


def save_pca_visualization(
    feature_path: Path,
    rgb_path: Path,
    mask_path: Path,
    destination: Path,
) -> None:
    """HWC NPYの前景特徴をPCAでRGB化し、crop RGBとの比較画像を保存する。

    Args:
        feature_path: ``(Hpatch, Wpatch, C)`` のfloat32特徴量。
        rgb_path: 同じ物体のcrop RGB。
        mask_path: 同じ物体のcrop mask。
        destination: 比較PNGの保存先。

    Raises:
        DataContractError: 特徴量の形式が不正、または前景patchが不足する場合。
    """
    features = np.load(feature_path, allow_pickle=False)
    if (
        features.ndim != FEATURE_NDIM
        or features.shape[2] < PCA_COMPONENTS
        or features.dtype != np.dtype(FEATURE_DTYPE)
    ):
        raise DataContractError("PCA可視化にはfloat32 HWC特徴量が必要である")

    grid_height, grid_width, channels = features.shape
    valid = _foreground_patches(mask_path, grid_height, grid_width)
    vectors = features.reshape(-1, channels)[valid]
    # PCAの3成分を作るには、最低でも3つの前景patchが必要である
    if len(vectors) < PCA_COMPONENTS:
        raise DataContractError(
            f"PCA可視化に必要な前景patchが{PCA_COMPONENTS}個未満である"
        )

    colors = _pca_colors(vectors)
    grid = np.zeros((grid_height * grid_width, PCA_COMPONENTS), dtype=np.uint8)
    grid[valid] = colors
    pca_image = Image.fromarray(
        grid.reshape(grid_height, grid_width, PCA_COMPONENTS), mode="RGB"
    )

    with Image.open(rgb_path) as source:
        rgb = source.convert("RGB")
    # patch格子は補間せず拡大し、patchごとの色の境目をそのまま見せる
    pca_image = pca_image.resize(rgb.size, Image.Resampling.NEAREST)
    comparison = Image.new("RGB", (rgb.width * 2, rgb.height), RGB_BACKGROUND)
    comparison.paste(rgb, (0, 0))
    comparison.paste(pca_image, (rgb.width, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(destination, format=IMAGE_FORMAT)


def _foreground_patches(
    mask_path: Path, grid_height: int, grid_width: int
) -> np.ndarray:
    """maskをpatch格子へ縮め、前景patchの真偽配列を返す。

    Args:
        mask_path: crop maskのパス。
        grid_height: 縦方向のpatch数。
        grid_width: 横方向のpatch数。

    Returns:
        長さ ``grid_height * grid_width`` の真偽配列。
    """
    with Image.open(mask_path) as source:
        small = source.convert("L").resize(
            (grid_width, grid_height), Image.Resampling.BILINEAR
        )
    coverage = np.asarray(small, dtype=np.float32).reshape(-1) / MASK_FOREGROUND
    # patchの一部だけmaskにかかる場合も前景として扱う閾値を使う
    return coverage > PCA_MASK_THRESHOLD


def _pca_colors(vectors: np.ndarray) -> np.ndarray:
    """前景patchのベクトルを、PCA上位3成分のRGB色に変換する。

    Args:
        vectors: ``(patch数, channels)`` の特徴ベクトル。

    Returns:
        ``(patch数, 3)`` のuint8色。
    """
    normalized = vectors / np.maximum(
        np.linalg.norm(vectors, axis=1, keepdims=True), NORM_EPSILON
    )
    centered = normalized - normalized.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ components[:PCA_COMPONENTS].T

    colors = np.zeros_like(projected)
    for index in range(PCA_COMPONENTS):
        component = projected[:, index]
        # SVDの符号は任意なので、最大寄与の要素が正になるよう揃えて色の再現性を保つ
        pivot = int(np.argmax(np.abs(components[index])))
        if components[index, pivot] < 0:
            component = -component
        low, high = float(component.min()), float(component.max())
        if high - low > RANGE_EPSILON:
            colors[:, index] = (component - low) / (high - low)
    scaled: np.ndarray = np.clip(colors * COLOR_MAX, 0, COLOR_MAX)
    return scaled.astype(np.uint8)
