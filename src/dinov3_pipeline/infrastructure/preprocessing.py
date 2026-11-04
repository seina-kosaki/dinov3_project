"""RGB・depth・maskに共通する固定サイズcropを、リサイズなしで生成する。

責務:
    3つの画像を同じ座標・同じ寸法で切り出し、mask外のRGBを0にする。
    depthは指定に応じてmask外を0にするか、crop内の生値を保つ。
    承認状態の検証やファイル出力は担わない。

処理の流れ:
    1. RGB・mask・depthを開き、寸法の一致を確認する。
    2. maskの前景範囲が収まる位置にcrop枠を決める（画像端では内側へ寄せる）。
    3. 同じ枠で3画像を切り出し、RGB背景を0にする。
    4. depthは設定に応じてmask背景を0にするか、生値のままにする。

関連ファイル:
    domain/models.py（ReadySample）、infrastructure/dinov3_extractor.py、
    infrastructure/delivery_exporter.py。

設計意図:
    resizeやpaddingは画素の補間・間引きを起こし、RGBとdepthの位置対応を崩す。
    そのため固定サイズの切り出しだけを行い、収まらない物体は縮めずに拒否する。
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from dinov3_pipeline.constants import DEPTH_BACKGROUND, MASK_FOREGROUND, RGB_BACKGROUND
from dinov3_pipeline.domain.models import DataContractError, ReadySample


@dataclass(frozen=True)
class CroppedInputs:
    """同じ座標から切り出したRGB・mask・depthと、元画像上のcrop範囲である。"""

    rgb: Image.Image
    mask: Image.Image
    depth: Image.Image
    crop_box: tuple[int, int, int, int]


def prepare_crop(
    sample: ReadySample, crop_size: int, mask_depth: bool = True
) -> CroppedInputs:
    """mask物体を含む固定サイズcropを、リサイズせずに作成する。

    Args:
        sample: RGB・mask・depthのパスを持つ承認済み物体。
        crop_size: 正方形cropの辺長（画素）。
        mask_depth: Trueならdepthのmask外を0にし、Falseならcrop内の生値を保つ。

    Returns:
        背景を0にしたRGB・二値mask・指定に応じて処理したdepthと、元画像座標のcrop範囲。

    Raises:
        DataContractError: 寸法不一致、空mask、またはcropに物体が収まらない場合。
    """
    rgb, raw_mask, depth = _open_modalities(sample)
    if crop_size > rgb.width or crop_size > rgb.height:
        raise DataContractError(f"crop_sizeが元画像より大きい: {sample.sample_id}")

    # 0より大きい画素を前景に統一し、JPEGなどの階調ずれを前景判定に持ち込まない
    mask = raw_mask.point(lambda value: MASK_FOREGROUND if value > 0 else 0)
    crop_box = _centered_crop_box(mask, rgb.size, crop_size, sample.sample_id)

    cropped_mask = mask.crop(crop_box)
    # 背景を0にして、CNNが対象物以外の画素や距離を使えないようにする
    cropped_rgb = Image.composite(
        rgb.crop(crop_box),
        Image.new("RGB", (crop_size, crop_size), RGB_BACKGROUND),
        cropped_mask,
    )
    cropped_depth = depth.crop(crop_box)
    if mask_depth:
        # CNNが対象物以外の距離を使わない設定では、mask外だけを0にする
        cropped_depth = Image.composite(
            cropped_depth,
            Image.new(depth.mode, (crop_size, crop_size), DEPTH_BACKGROUND),
            cropped_mask,
        )
    return CroppedInputs(cropped_rgb, cropped_mask, cropped_depth, crop_box)


def _open_modalities(
    sample: ReadySample,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """RGB・mask・depthを読み込み、寸法の一致を確認する。

    Args:
        sample: 承認済み物体。

    Returns:
        RGB、8bit mask、元の値を保つdepth。

    Raises:
        DataContractError: 開けない、または寸法が一致しない場合。
    """
    try:
        with Image.open(sample.rgb_path) as source:
            rgb = source.convert("RGB")
        with Image.open(sample.mask_path) as source:
            mask = source.convert("L")
        with Image.open(sample.depth_path) as source:
            # depthは値を変えないよう、変換せずコピーだけ取る
            depth = source.copy()
    except OSError as error:
        raise DataContractError(
            f"RGB・mask・depthを開けない: {sample.sample_id}"
        ) from error

    # 寸法が違うと同じ座標で切り出しても画素が対応しないため拒否する
    if not rgb.size == mask.size == depth.size:
        raise DataContractError(
            f"RGB・mask・depthの寸法が一致しない: {sample.sample_id}"
        )
    return rgb, mask, depth


def _centered_crop_box(
    mask: Image.Image,
    image_size: tuple[int, int],
    crop_size: int,
    sample_id: str,
) -> tuple[int, int, int, int]:
    """mask前景の中心に合わせ、画像内に収まるcrop枠を決める。

    Args:
        mask: 二値mask。
        image_size: 元画像の（幅, 高さ）。
        crop_size: 正方形cropの辺長。
        sample_id: エラー表示用のID。

    Returns:
        （左, 上, 右, 下）のcrop枠。

    Raises:
        DataContractError: 前景が無い、またはcropより大きい場合。
    """
    bounds = mask.getbbox()
    if bounds is None:
        raise DataContractError(f"maskに前景画素がない: {sample_id}")
    left_bound, top_bound, right_bound, bottom_bound = bounds
    # 物体を縮めると画素間隔が変わるため、収まらない物体は拒否する
    if right_bound - left_bound > crop_size or bottom_bound - top_bound > crop_size:
        raise DataContractError(
            f"対象maskが設定crop_sizeに収まらない: {sample_id}, "
            f"crop_size={crop_size}, foreground_width={right_bound - left_bound}, "
            f"foreground_height={bottom_bound - top_bound}"
        )

    width, height = image_size
    # 前景の中心に合わせたうえで、画像端では内側へ寄せて枠を画像内に保つ
    left = min(max((left_bound + right_bound - crop_size) // 2, 0), width - crop_size)
    top = min(max((top_bound + bottom_bound - crop_size) // 2, 0), height - crop_size)
    return left, top, left + crop_size, top + crop_size
