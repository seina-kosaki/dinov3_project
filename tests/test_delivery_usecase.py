"""抽出から公開までの流れ（usecase）を、ダミー抽出器で確認する。

責務:
    run_delivery が、承認済み物体を抽出・再検証し、失効した物体を出力しないことを確認する。

処理の流れ:
    実際のsession DBとRepositoryを使い、モデルだけをダミー抽出器に差し替えて実行する。

関連ファイル:
    src/dinov3_pipeline/usecase/delivery.py、src/dinov3_pipeline/domain/ports.py、tests/conftest.py。

設計意図:
    GPUやHugging Face認証を使わずに、抽出前後の承認再確認という安全契約を確認する。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from dinov3_pipeline.domain.models import (
    DataContractError,
    FeatureArtifact,
    ReadySample,
    SampleLoadResult,
)
from dinov3_pipeline.infrastructure.session_repository import SessionSam3Repository
from dinov3_pipeline.usecase.delivery import run_delivery

CROP_SIZE = 64
VERSION = "capture_v001"
TIMEOUT_SECONDS = 1.0


class ArrayExtractor:
    """小さなHWC特徴量を保存するダミー抽出器である。

    Args:
        after_write: 保存直後に実行する処理（抽出中の外部変更を再現する）。
    """

    def __init__(self, after_write: Callable[[], None] | None = None) -> None:
        self.after_write = after_write

    def extract(self, sample: ReadySample, destination: Path) -> FeatureArtifact:
        """固定のHWC特徴量を保存し、artifactを返す。

        Args:
            sample: 承認済み物体（使用しない）。
            destination: 保存先。

        Returns:
            保存した特徴量のartifact。
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            destination,
            np.arange(16, dtype=np.float32).reshape(2, 2, 4),
            allow_pickle=False,
        )
        if self.after_write is not None:
            self.after_write()
        return FeatureArtifact(
            destination, (2, 2, 4), "float32", "dinov3_test", "weights-1"
        )


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


def test_run_delivery_ready_sample_completes_all_samples(
    session_root: tuple[Path, Path], tmp_path: Path
) -> None:
    """承認済み物体を全件完了する。"""
    # 準備
    root, _ = session_root
    repository = SessionSam3Repository(root, TIMEOUT_SECONDS)

    # 実行
    summary = run_delivery(
        repository.load_ready_samples(),
        repository,
        ArrayExtractor(),
        tmp_path / "delivery",
        CROP_SIZE,
        VERSION,
    )

    # 検証
    assert (
        summary.eligible_count,
        summary.completed_count,
        summary.failed_sample_ids,
    ) == (1, 1, ())


def test_run_delivery_ready_sample_publishes_cnn_metadata(
    session_root: tuple[Path, Path], tmp_path: Path
) -> None:
    """CNN bundleのmetadata SQLiteを公開する。"""
    # 準備
    root, _ = session_root
    repository = SessionSam3Repository(root, TIMEOUT_SECONDS)

    # 実行
    summary = run_delivery(
        repository.load_ready_samples(),
        repository,
        ArrayExtractor(),
        tmp_path / "delivery",
        CROP_SIZE,
        VERSION,
    )

    # 検証
    assert (summary.output.cnn_path / "metadata.sqlite3").is_file()


def test_run_delivery_revoked_during_extraction_raises_without_publishing(
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
        run_delivery(
            repository.load_ready_samples(),
            repository,
            extractor,
            delivery_root,
            CROP_SIZE,
            VERSION,
        )
    assert not (delivery_root / "cnn").exists()


def test_run_delivery_no_samples_raises_contract_error(
    session_root: tuple[Path, Path], tmp_path: Path
) -> None:
    """承認済み物体が無い場合は拒否する。"""
    # 準備
    repository = SessionSam3Repository(session_root[0], TIMEOUT_SECONDS)

    # 実行・検証
    with pytest.raises(DataContractError, match="有効なsampleがない"):
        run_delivery(
            SampleLoadResult((), ()),
            repository,
            ArrayExtractor(),
            tmp_path / "delivery",
            CROP_SIZE,
            VERSION,
        )
