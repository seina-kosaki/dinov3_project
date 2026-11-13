"""合成画像をDINOv3で推論し、特徴量のPCA可視化を保存する。"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from dinov3_pipeline.dinov3.extractor import Dinov3FeatureExtractor
from dinov3_pipeline.saving.pca_visualizer import save_pca_visualization
from dinov3_pipeline.settings import load_settings

LOGGER = logging.getLogger(__name__)
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output"
INPUT_IMAGE_PATH = Path(r"E:\医療分野\datasets\SAM3\labeled_surgical_tools\20260910\train\verified\straight_dissection_clamp\raw") / "separado37.png"
PATCH_SIZE = 16
FEATURE_KEYS = (
    "feature_block_1",
    "feature_block_2",
    "feature_block_3",
    "feature_block_4",
    "feature",
)


def main() -> None:
    """指定画像のDINOv3推論とPCA画像出力を順に実行する。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    mask_path = OUTPUT_DIRECTORY / "input_mask.png"
    pca_path = OUTPUT_DIRECTORY / "pca_comparison.png"

    with Image.open(INPUT_IMAGE_PATH) as source:
        width = source.width // PATCH_SIZE * PATCH_SIZE
        height = source.height // PATCH_SIZE * PATCH_SIZE
        if width == 0 or height == 0:
            raise ValueError(f"画像の縦横は{PATCH_SIZE}ピクセル以上必要です: {source.size}")
        rgb_image = source.convert("RGB")
        if rgb_image.size != (width, height):
            rgb_image = rgb_image.resize((width, height), Image.Resampling.LANCZOS)
    Image.new("L", rgb_image.size, 255).save(mask_path)

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
            INPUT_IMAGE_PATH,
            mask_path,
            pca_path,
            mask_rgb_background=True,
        )

    LOGGER.info("PCA比較画像を保存しました: %s", pca_path)


if __name__ == "__main__":
    main()
