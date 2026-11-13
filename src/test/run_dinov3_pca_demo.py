"""合成画像をDINOv3で推論し、特徴量のPCA可視化を保存する。"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dinov3_pipeline.dinov3.extractor import Dinov3FeatureExtractor
from dinov3_pipeline.saving.pca_visualizer import save_pca_visualization
from dinov3_pipeline.settings import load_settings

LOGGER = logging.getLogger(__name__)
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output"
FEATURE_KEYS = (
    "feature_block_1",
    "feature_block_2",
    "feature_block_3",
    "feature_block_4",
    "feature",
)


def _create_dummy_image(size: int, destination: Path) -> None:
    """Pillowで色や形状に変化のあるテスト画像を作成する。"""
    image = Image.new("RGB", (size, size), (24, 36, 56))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size // 2, size // 2), fill=(220, 70, 50))
    draw.ellipse((size // 3, size // 3, size - 1, size - 1), fill=(40, 180, 120))
    draw.line((0, size - 1, size - 1, 0), fill=(240, 220, 60), width=max(1, size // 32))
    image.save(destination)


def main() -> None:
    """画像生成、DINOv3推論、PCA画像出力を順に実行する。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    rgb_path = OUTPUT_DIRECTORY / "dummy_rgb.png"
    mask_path = OUTPUT_DIRECTORY / "dummy_mask.png"
    pca_path = OUTPUT_DIRECTORY / "pca_comparison.png"
    _create_dummy_image(settings.crop_size, rgb_path)
    Image.new("L", (settings.crop_size, settings.crop_size), 255).save(mask_path)

    with Image.open(rgb_path) as source:
        rgb_image = source.convert("RGB")

    extractor = Dinov3FeatureExtractor(
        settings.model_name,
        settings.pretrain_dataset,
        settings.device,
    )
    features = extractor.infer(rgb_image)
    for key, feature in features.items():
        LOGGER.info("%s: shape=%s dtype=%s", key, feature.shape, feature.dtype)

    with tempfile.TemporaryDirectory(prefix="dinov3-pca-demo-") as temporary_directory:
        feature_paths = []
        for key in FEATURE_KEYS:
            feature_path = Path(temporary_directory) / f"{key}.npy"
            np.save(feature_path, features[key], allow_pickle=False)
            feature_paths.append(feature_path)

        save_pca_visualization(
            tuple(feature_paths),
            rgb_path,
            mask_path,
            pca_path,
            mask_rgb_background=True,
        )

    LOGGER.info("PCA比較画像を保存しました: %s", pca_path)


if __name__ == "__main__":
    main()
