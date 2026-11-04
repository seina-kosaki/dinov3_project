"""固定サイズcropが画素と位置対応を保つことを検証する。

責務:
    prepare_crop の座標・寸法・depth値・背景0化・拒否条件を確認する。

処理の流れ:
    conftest の make_sample で実ファイルを作り、prepare_crop を呼んで結果を検証する。

関連ファイル:
    src/dinov3_pipeline/infrastructure/preprocessing.py、tests/conftest.py。

設計意図:
    resizeなしでRGB・depth・maskが同じ座標になる契約を、1観点ずつ確認する。
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import DEPTH_VALUE, RGB_COLOR, SampleFactory
from PIL import Image

from dinov3_pipeline.domain.models import DataContractError
from dinov3_pipeline.infrastructure.preprocessing import prepare_crop

CROP_SIZE = 32


def test_prepare_crop_object_inside_image_returns_box_centered_on_mask(
    make_sample: SampleFactory,
) -> None:
    """mask前景の中心に合わせたcrop枠を返す。"""
    # 準備
    sample = make_sample()

    # 実行
    cropped = prepare_crop(sample, CROP_SIZE)

    # 検証
    assert cropped.crop_box == (28, 24, 60, 56)


def test_prepare_crop_object_near_edge_keeps_box_inside_image(
    make_sample: SampleFactory,
) -> None:
    """画像端の物体でも、crop枠を画像内に収める。"""
    # 準備
    sample = make_sample(mask_box=(0, 0, 8, 8))

    # 実行
    cropped = prepare_crop(sample, CROP_SIZE)

    # 検証
    assert cropped.crop_box == (0, 0, CROP_SIZE, CROP_SIZE)


def test_prepare_crop_valid_input_returns_same_size_for_all_modalities(
    make_sample: SampleFactory,
) -> None:
    """RGB・mask・depthを同じ寸法で返し、リサイズしない。"""
    # 準備
    sample = make_sample()

    # 実行
    cropped = prepare_crop(sample, CROP_SIZE)

    # 検証
    assert (
        cropped.rgb.size
        == cropped.mask.size
        == cropped.depth.size
        == (CROP_SIZE, CROP_SIZE)
    )


def test_prepare_crop_valid_input_preserves_float_depth_inside_mask(
    make_sample: SampleFactory,
) -> None:
    """mask内のdepthは元のfloat値のまま保つ。"""
    # 準備
    sample = make_sample()

    # 実行
    cropped = prepare_crop(sample, CROP_SIZE)

    # 検証
    center = (CROP_SIZE // 2, CROP_SIZE // 2)
    assert cropped.depth.mode == "F"
    assert cropped.depth.getpixel(center) == pytest.approx(DEPTH_VALUE)


def test_prepare_crop_valid_input_keeps_rgb_color_inside_mask(
    make_sample: SampleFactory,
) -> None:
    """mask内のRGBは元の色のまま保つ。"""
    # 準備
    sample = make_sample()

    # 実行
    cropped = prepare_crop(sample, CROP_SIZE)

    # 検証
    assert cropped.rgb.getpixel((CROP_SIZE // 2, CROP_SIZE // 2)) == RGB_COLOR


@pytest.mark.parametrize("modality", ["rgb", "depth"])
def test_prepare_crop_valid_input_zeroes_background(
    make_sample: SampleFactory, modality: str
) -> None:
    """mask外のRGBとdepthを0にする。"""
    # 準備
    sample = make_sample()

    # 実行
    cropped = prepare_crop(sample, CROP_SIZE)

    # 検証
    background = np.asarray(getattr(cropped, modality))[0, 0]
    assert np.all(background == 0)


@pytest.mark.parametrize("mask_box", [(0, 0, 64, 16), (0, 0, 16, 64), (0, 0, 64, 64)])
def test_prepare_crop_object_larger_than_crop_raises_without_resizing(
    make_sample: SampleFactory,
    mask_box: tuple[int, int, int, int],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """crop寸法を超える物体は、縮めずに拒否する。"""
    # 準備
    sample = make_sample(mask_box=mask_box)

    # 実行・検証
    with pytest.raises(DataContractError, match="収まらない") as caught:
        prepare_crop(sample, CROP_SIZE)
    assert str(caught.value) == (
        f"対象maskが設定crop_sizeに収まらない: {sample.sample_id}, "
        f"crop_size={CROP_SIZE}, foreground_width={mask_box[2]}, "
        f"foreground_height={mask_box[3]}"
    )
    assert capsys.readouterr() == ("", "")
    assert not caplog.records


def test_prepare_crop_empty_mask_raises_contract_error(
    make_sample: SampleFactory,
) -> None:
    """前景が無いmaskを拒否する。"""
    # 準備
    sample = make_sample()
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(sample.mask_path)

    # 実行・検証
    with pytest.raises(DataContractError, match="前景画素がない"):
        prepare_crop(sample, CROP_SIZE)


def test_prepare_crop_size_mismatch_raises_contract_error(
    make_sample: SampleFactory,
) -> None:
    """RGBとmaskの寸法が違う入力を拒否する。"""
    # 準備
    sample = make_sample()
    Image.new("L", (32, 32), 255).save(sample.mask_path)

    # 実行・検証
    with pytest.raises(DataContractError, match="寸法が一致しない"):
        prepare_crop(sample, CROP_SIZE)
