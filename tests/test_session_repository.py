"""session DBからの承認済み物体の取得と再検証を確認する。

責務:
    SessionSam3Repository が ready 行だけを返し、書き込まず、取消や改変を検出することを確認する。

処理の流れ:
    conftest の session_root でDBを作り、読込・再検証を呼んで結果を検証する。

関連ファイル:
    src/dinov3_pipeline/infrastructure/session_repository.py、tests/conftest.py。

設計意図:
    承認の取消・原本の改変が出力に紛れ込まないという、中核の安全契約を守る。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from dinov3_pipeline.common.models import DataContractError
from dinov3_pipeline.data_loading.session_repository import SessionSam3Repository

TIMEOUT_SECONDS = 1.0
DB_OBJECT_COUNT_SQL = "SELECT COUNT(*) FROM sqlite_master"


def _execute(database_path: Path, statement: str) -> None:
    """テスト用に、session DBへ変更を加える。

    Args:
        database_path: 対象のDB。
        statement: 実行するSQL。
    """
    with sqlite3.connect(database_path) as connection:
        connection.execute(statement)


def test_load_ready_samples_mixed_rows_returns_ready_row_only(
    session_root: tuple[Path, Path],
) -> None:
    """is_dino_ready=1の行だけを返す。"""
    # 準備
    root, _ = session_root

    # 実行
    loaded = SessionSam3Repository(root, TIMEOUT_SECONDS).load_ready_samples()

    # 検証
    assert [sample.image_id for sample in loaded.samples] == ["image-001"]


def test_load_ready_samples_ready_row_returns_capture_date_from_directory(
    session_root: tuple[Path, Path],
) -> None:
    """日付ディレクトリ名を撮影日として返す。"""
    # 準備
    root, _ = session_root

    # 実行
    loaded = SessionSam3Repository(root, TIMEOUT_SECONDS).load_ready_samples()

    # 検証
    assert loaded.samples[0].capture_date == "2026-10-30"


def test_load_ready_samples_any_input_does_not_change_source_database(
    session_root: tuple[Path, Path],
) -> None:
    """撮影側DBへテーブルを追加しない。"""
    # 準備
    root, database_path = session_root
    with sqlite3.connect(database_path) as connection:
        before = connection.execute(DB_OBJECT_COUNT_SQL).fetchone()[0]

    # 実行
    SessionSam3Repository(root, TIMEOUT_SECONDS).load_ready_samples()

    # 検証
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(DB_OBJECT_COUNT_SQL).fetchone()[0] == before


def test_load_ready_samples_missing_depth_path_reports_issue(
    session_root: tuple[Path, Path],
) -> None:
    """depthのパスが無い行は、除外して理由を返す。"""
    # 準備
    root, database_path = session_root
    _execute(
        database_path,
        "UPDATE sam3_dino_results SET depth_path = '' WHERE is_dino_ready = 1",
    )

    # 実行
    loaded = SessionSam3Repository(root, TIMEOUT_SECONDS).load_ready_samples()

    # 検証
    assert loaded.samples == () and "depth" in loaded.issues[0]


def test_load_ready_samples_no_database_raises_contract_error(tmp_path: Path) -> None:
    """session DBが1つも無い共有rootを拒否する。"""
    # 準備
    repository = SessionSam3Repository(tmp_path, TIMEOUT_SECONDS)

    # 実行・検証
    with pytest.raises(DataContractError, match="見つからない"):
        repository.load_ready_samples()


def test_validate_sample_unchanged_input_passes(
    session_root: tuple[Path, Path],
) -> None:
    """変更の無い承認済み物体の検証は通る。"""
    # 準備
    repository = SessionSam3Repository(session_root[0], TIMEOUT_SECONDS)
    sample = repository.load_ready_samples().samples[0]

    # 実行・検証
    repository.validate_sample(sample)


def test_validate_sample_ready_flag_revoked_raises_contract_error(
    session_root: tuple[Path, Path],
) -> None:
    """読込後に承認が取り消された物体を拒否する。"""
    # 準備
    root, database_path = session_root
    repository = SessionSam3Repository(root, TIMEOUT_SECONDS)
    sample = repository.load_ready_samples().samples[0]
    _execute(database_path, "UPDATE sam3_dino_results SET is_dino_ready = 0")

    # 実行・検証
    with pytest.raises(DataContractError, match="変更または失効"):
        repository.validate_sample(sample)


def test_validate_sample_mask_file_modified_raises_contract_error(
    session_root: tuple[Path, Path],
) -> None:
    """読込後にmaskが書き換えられた物体を拒否する。"""
    # 準備
    repository = SessionSam3Repository(session_root[0], TIMEOUT_SECONDS)
    sample = repository.load_ready_samples().samples[0]
    Image.new("L", (64, 64), 128).save(sample.mask_path)

    # 実行・検証
    with pytest.raises(DataContractError, match="maskのhash"):
        repository.validate_sample(sample)
