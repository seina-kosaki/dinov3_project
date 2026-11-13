"""抽出から公開までの流れ（usecase）を、ダミー抽出器で確認する。

責務:
    process_approved_samples が、承認済み物体を抽出・再検証し、失効した物体を出力しないことを確認する。

処理の流れ:
    実際のsession DBとRepositoryを使い、モデルだけをダミー抽出器に差し替えて実行する。

関連ファイル:
    src/dinov3_pipeline/usecase/sample_processing.py、src/dinov3_pipeline/domain/ports.py、tests/conftest.py。

設計意図:
    GPUやHugging Face認証を使わずに、抽出前後の承認再確認という安全契約を確認する。
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from dinov3_pipeline.common.models import (
    DataContractError,
    ReadySample,
    SampleLoadResult,
)
from dinov3_pipeline.data_loading.preprocessing import prepare_crop
from dinov3_pipeline.data_loading.session_repository import SessionSam3Repository
from dinov3_pipeline.usecase.sample_processing import process_approved_samples

CROP_SIZE = 64
VERSION = "capture_v001"
TIMEOUT_SECONDS = 1.0


class ArrayExtractor:
    """小さなHWC特徴量を保存するダミー抽出器である。

    Args:
        after_write: 保存直後に実行する処理（抽出中の外部変更を再現する）。
    """

    def __init__(self, after_write: Callable[[], None] | None = None) -> None:
        """特徴量生成後に実行する任意の処理を保持する。"""
        self.after_write = after_write
        self.model_name = "dinov3_test"
        self.last_rgb = None
        self.last_mask = None

    def infer(self, rgb_image: object, mask_image: object = None) -> dict[str, np.ndarray]:
        """固定のHWC特徴量を返す。"""
        self.last_rgb = rgb_image
        self.last_mask = mask_image
        features = {"feature": np.arange(16, dtype=np.float32).reshape(2, 2, 4)}
        for index in range(1, 5):
            features[f"feature_block_{index}"] = np.full((2, 2, 4), index, dtype=np.float32)
        if self.after_write is not None:
            self.after_write()
        return features


def _revoke(database_path: Path) -> Callable[[], None]:
    """承認を取り消す処理を返す。

    Args:
        database_path: 対象のsession DB。

    Returns:
        呼ぶとready状態を外す関数。
    """

    def revoke() -> None:
        """is_dino_ready を0にする。"""
        with sqlite3.connect(database_path) as connection:
            connection.execute("UPDATE sam3_dino_results SET is_dino_ready = 0")

    return revoke


def test_process_approved_samples_ready_sample_completes_all_samples(
    session_root: tuple[Path, Path], tmp_path: Path
) -> None:
    """承認済み物体を全件完了する。"""
    # 準備
    root, _ = session_root
    repository = SessionSam3Repository(root, TIMEOUT_SECONDS)

    # 実行
    summary = process_approved_samples(
        repository.load_ready_samples(),
        repository,
        ArrayExtractor(),
        tmp_path / "delivery",
        CROP_SIZE,
        dataset_version=VERSION,
    )

    # 検証
    assert (
        summary.eligible_count,
        summary.completed_count,
        summary.failed_sample_ids,
    ) == (1, 1, ())


def test_process_approved_samples_ready_sample_publishes_cnn_metadata(
    session_root: tuple[Path, Path], tmp_path: Path
) -> None:
    """CNN bundleのmetadata SQLiteを公開する。"""
    # 準備
    root, _ = session_root
    repository = SessionSam3Repository(root, TIMEOUT_SECONDS)

    # 実行
    summary = process_approved_samples(
        repository.load_ready_samples(),
        repository,
        ArrayExtractor(),
        tmp_path / "delivery",
        CROP_SIZE,
        dataset_version=VERSION,
    )

    # 検証
    assert (summary.output.cnn_path / "metadata.sqlite3").is_file()


@pytest.mark.parametrize("mask_rgb_background", [False, True])
def test_process_passes_prepared_rgb_and_optional_mask_to_inference(
    session_root: tuple[Path, Path],
    tmp_path: Path,
    mask_rgb_background: bool,
) -> None:
    """整形済みRGBと設定に応じたmaskを推論器へ渡す。"""
    repository = SessionSam3Repository(session_root[0], TIMEOUT_SECONDS)
    extractor = ArrayExtractor()
    process_approved_samples(
        repository.load_ready_samples(),
        repository,
        extractor,
        tmp_path / "delivery",
        CROP_SIZE,
        dataset_version=VERSION,
        mask_rgb_background=mask_rgb_background,
    )
    assert extractor.last_rgb is not None
    assert extractor.last_rgb.size == (CROP_SIZE, CROP_SIZE)
    if mask_rgb_background:
        assert extractor.last_mask is not None
        assert extractor.last_mask.size == (CROP_SIZE, CROP_SIZE)
    else:
        assert extractor.last_mask is None


def test_process_approved_samples_revoked_during_extraction_raises_without_publishing(
    session_root: tuple[Path, Path], tmp_path: Path
) -> None:
    """抽出中に承認が取り消された物体は、出力せずに拒否する。"""
    # 準備
    root, database_path = session_root
    repository = SessionSam3Repository(root, TIMEOUT_SECONDS)
    extractor = ArrayExtractor(after_write=_revoke(database_path))
    delivery_root = tmp_path / "delivery"

    # 実行・検証
    with pytest.raises(DataContractError, match="全てのready sample"):
        process_approved_samples(
            repository.load_ready_samples(),
            repository,
            extractor,
            delivery_root,
            CROP_SIZE,
            dataset_version=VERSION,
        )
    assert not (delivery_root / "cnn").exists()


def test_process_approved_samples_no_samples_raises_contract_error(
    session_root: tuple[Path, Path], tmp_path: Path
) -> None:
    """承認済み物体が無い場合は拒否する。"""
    # 準備
    repository = SessionSam3Repository(session_root[0], TIMEOUT_SECONDS)

    # 実行・検証
    with pytest.raises(DataContractError, match="有効なsampleがない"):
        process_approved_samples(
            SampleLoadResult((), ()),
            repository,
            ArrayExtractor(),
            tmp_path / "delivery",
            CROP_SIZE,
            dataset_version=VERSION,
        )


def test_process_approved_samples_crop_failure_logs_once_and_excludes_sample(
    session_root: tuple[Path, Path],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """crop超過の診断は既存の抽出境界で一度通知し、公開しない。"""
    repository = SessionSam3Repository(session_root[0], TIMEOUT_SECONDS)
    crop_size = 32

    with caplog.at_level(logging.ERROR), pytest.raises(DataContractError, match="全てのready sample"):
        process_approved_samples(
            repository.load_ready_samples(), repository, ArrayExtractor(),
            tmp_path / "delivery", crop_size, dataset_version=VERSION,
        )
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.name == "dinov3_pipeline.usecase.sample_processing"
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], DataContractError)
    assert "crop_size=32, foreground_width=64, foreground_height=64" in caplog.text
    assert capsys.readouterr() == ("", "")
    assert not (tmp_path / "delivery" / "cnn").exists()
