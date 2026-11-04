"""patch tokenがHWC形式で保存される契約を検証する。

責務:
    patch_tokens_to_hwc の並び順と、不正なtoken数の拒否を確認する。

処理の流れ:
    小さなtoken行列を作り、変換結果のshape・dtype・位置を検証する。

関連ファイル:
    src/dinov3_pipeline/infrastructure/dinov3_extractor.py。

設計意図:
    空間位置のずれはCNN側で気づきにくいため、並び順を単体で確認する。
"""

from __future__ import annotations

import numpy as np
import pytest

from dinov3_pipeline.domain.models import DataContractError
from dinov3_pipeline.infrastructure.dinov3_extractor import patch_tokens_to_hwc


def test_patch_tokens_to_hwc_valid_grid_returns_float32_hwc_shape() -> None:
    """有効なgridを、float32のHWC形状に変換する。"""
    # 準備
    tokens = np.arange(12, dtype=np.float64).reshape(4, 3)

    # 実行
    result = patch_tokens_to_hwc(tokens, grid_height=2, grid_width=2)

    # 検証
    assert result.shape == (2, 2, 3)
    assert result.dtype == np.float32


def test_patch_tokens_to_hwc_valid_grid_preserves_row_major_order() -> None:
    """行優先の並びを保ち、空間位置をずらさない。"""
    # 準備
    tokens = np.arange(12, dtype=np.float64).reshape(4, 3)

    # 実行
    result = patch_tokens_to_hwc(tokens, grid_height=2, grid_width=2)

    # 検証
    np.testing.assert_array_equal(result[1, 0], np.array([6, 7, 8], dtype=np.float32))


def test_patch_tokens_to_hwc_token_count_mismatch_raises_contract_error() -> None:
    """gridと合わないtoken数を拒否する。"""
    # 準備
    tokens = np.zeros((3, 4), dtype=np.float32)

    # 実行・検証
    with pytest.raises(DataContractError):
        patch_tokens_to_hwc(tokens, grid_height=2, grid_width=2)
