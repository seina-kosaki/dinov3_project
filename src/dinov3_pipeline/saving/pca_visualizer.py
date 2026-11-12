"""PCA-RGBのdebug可視化を生成する。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from dinov3_pipeline.common.constants import (
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
from dinov3_pipeline.common.models import DataContractError


def save_pca_visualization(
    feature_paths: tuple[Path, ...],
    rgb_path: Path,
    mask_path: Path,
    destination: Path,
    mask_rgb_background: bool = True,
) -> None:
    """特徴量の PCA 主成分を RGB と並べてデバッグ画像として保存する。"""
    with Image.open(rgb_path) as source:
        rgb = source.convert("RGB")

    panels = [rgb]
    for feature_path in feature_paths:
        features = np.load(feature_path, allow_pickle=False)
        if (
            features.ndim != FEATURE_NDIM
            or features.shape[2] < PCA_COMPONENTS
            or features.dtype != np.dtype(FEATURE_DTYPE)
        ):
            raise DataContractError("PCA可視化にはfloat32 HWC特徴量が必要である")

        grid_height, grid_width, channels = features.shape
        if mask_rgb_background:
            valid = _foreground_patches(mask_path, grid_height, grid_width)
        else:
            valid = np.ones(grid_height * grid_width, dtype=bool)
        vectors = features.reshape(-1, channels)[valid]
        if len(vectors) < PCA_COMPONENTS:
            label = "前景patch" if mask_rgb_background else "patch"
            raise DataContractError(f"PCA可視化に必要な{label}が{PCA_COMPONENTS}個未満である")

        colors = _pca_colors(vectors)
        grid = np.zeros((grid_height * grid_width, PCA_COMPONENTS), dtype=np.uint8)
        grid[valid] = colors
        pca_image = Image.fromarray(grid.reshape(grid_height, grid_width, PCA_COMPONENTS), mode="RGB")
        panels.append(pca_image.resize(rgb.size, Image.Resampling.NEAREST))

    comparison = Image.new("RGB", (rgb.width * 3, rgb.height * 2), RGB_BACKGROUND)
    for index, panel in enumerate(panels):
        comparison.paste(panel, ((index % 3) * rgb.width, (index // 3) * rgb.height))
    destination.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(destination, format=IMAGE_FORMAT)


def _foreground_patches(mask_path: Path, grid_height: int, grid_width: int) -> np.ndarray:
    """crop の各 patch に対して、前景に相当する位置を 0/1 のmaskで返す。"""
    with Image.open(mask_path) as source:
        small = source.convert("L").resize((grid_width, grid_height), Image.Resampling.BILINEAR)
    coverage = np.asarray(small, dtype=np.float32).reshape(-1) / MASK_FOREGROUND
    return coverage > PCA_MASK_THRESHOLD


def _pca_colors(vectors: np.ndarray) -> np.ndarray:
    """PCA で圧縮した特徴量を RGB の色値へ変換する。"""
    normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), NORM_EPSILON)
    centered = normalized - normalized.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ components[:PCA_COMPONENTS].T

    colors = np.zeros_like(projected)
    for index in range(PCA_COMPONENTS):
        component = projected[:, index]
        pivot = int(np.argmax(np.abs(components[index])))
        if components[index, pivot] < 0:
            component = -component
        low, high = float(component.min()), float(component.max())
        if high - low > RANGE_EPSILON:
            colors[:, index] = (component - low) / (high - low)
    scaled: np.ndarray = np.clip(colors * COLOR_MAX, 0, COLOR_MAX)
    return scaled.astype(np.uint8)
