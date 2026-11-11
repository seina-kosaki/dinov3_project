"""RGB・depth・maskのcrop処理を定義する。"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from dinov3_pipeline.common.constants import DEPTH_BACKGROUND, MASK_FOREGROUND, RGB_BACKGROUND
from dinov3_pipeline.common.models import DataContractError, ReadySample


@dataclass(frozen=True)
class CroppedInputs:
    rgb: Image.Image
    mask: Image.Image
    depth: Image.Image
    crop_box: tuple[int, int, int, int]


def prepare_crop(sample: ReadySample, crop_size: int, mask_depth: bool = True) -> CroppedInputs:
    rgb, raw_mask, depth = _open_modalities(sample)
    if crop_size > rgb.width or crop_size > rgb.height:
        raise DataContractError(f"crop_sizeが元画像より大きい: {sample.sample_id}")

    mask = raw_mask.point(lambda value: MASK_FOREGROUND if value > 0 else 0)
    crop_box = _centered_crop_box(mask, rgb.size, crop_size, sample.sample_id)

    cropped_mask = mask.crop(crop_box)
    cropped_rgb = Image.composite(
        rgb.crop(crop_box),
        Image.new("RGB", (crop_size, crop_size), RGB_BACKGROUND),
        cropped_mask,
    )
    cropped_depth = depth.crop(crop_box)
    if mask_depth:
        cropped_depth = Image.composite(
            cropped_depth,
            Image.new(depth.mode, (crop_size, crop_size), DEPTH_BACKGROUND),
            cropped_mask,
        )
    return CroppedInputs(cropped_rgb, cropped_mask, cropped_depth, crop_box)


def _open_modalities(sample: ReadySample) -> tuple[Image.Image, Image.Image, Image.Image]:
    try:
        with Image.open(sample.rgb_path) as source:
            rgb = source.convert("RGB")
        with Image.open(sample.mask_path) as source:
            mask = source.convert("L")
        with Image.open(sample.depth_path) as source:
            depth = source.copy()
    except OSError as error:
        raise DataContractError(f"RGB・mask・depthを開けない: {sample.sample_id}") from error
    if not rgb.size == mask.size == depth.size:
        raise DataContractError(f"RGB・mask・depthの寸法が一致しない: {sample.sample_id}")
    return rgb, mask, depth


def _centered_crop_box(
    mask: Image.Image,
    image_size: tuple[int, int],
    crop_size: int,
    sample_id: str,
) -> tuple[int, int, int, int]:
    bounds = mask.getbbox()
    if bounds is None:
        raise DataContractError(f"maskに前景画素がない: {sample_id}")
    left_bound, top_bound, right_bound, bottom_bound = bounds
    if right_bound - left_bound > crop_size or bottom_bound - top_bound > crop_size:
        raise DataContractError(
            f"対象maskが設定crop_sizeに収まらない: {sample_id}, "
            f"crop_size={crop_size}, foreground_width={right_bound - left_bound}, "
            f"foreground_height={bottom_bound - top_bound}"
        )
    width, height = image_size
    left = min(max((left_bound + right_bound - crop_size) // 2, 0), width - crop_size)
    top = min(max((top_bound + bottom_bound - crop_size) // 2, 0), height - crop_size)
    return left, top, left + crop_size, top + crop_size
