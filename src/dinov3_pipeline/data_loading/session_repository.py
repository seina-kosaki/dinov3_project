"""session DBから承認済みsampleを読み込む。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from dinov3_pipeline.common.constants import (
    READY_TABLE,
    REQUIRED_READY_COLUMNS,
    SAMPLE_ID_PREFIX,
    SESSION_DB_GLOB,
)
from dinov3_pipeline.common.file_hash import sha256_of_file
from dinov3_pipeline.common.models import DataContractError, ReadySample, SampleLoadResult

READY_QUERY = f"SELECT * FROM {READY_TABLE} WHERE is_dino_ready = 1 ORDER BY session_id, image_id, revision, object_index"
REVALIDATE_QUERY = f"SELECT * FROM {READY_TABLE} WHERE session_id = ? AND image_id = ? AND prediction_id = ? AND revision = ? AND object_index = ? AND is_dino_ready = 1"


class SessionSam3Repository:
    """shared_data 配下の session SQLite を読み込み、承認済みサンプルを安全に抽出する。"""

    def __init__(self, data_root: Path, timeout_seconds: float) -> None:
        """共有データの root と読み取り用 timeout を保持する。"""
        self.data_root = Path(data_root).resolve()
        self.timeout_seconds = timeout_seconds

    def load_ready_samples(self) -> SampleLoadResult:
        """全session DBから承認済みsampleを集め、読み取り契約を検証する。

        Returns:
            検証済みsampleと処理対象外のissue一覧。

        Raises:
            DataContractError: 対象rootにsession metadata.sqlite3がない場合。
        """
        database_paths = sorted(self.data_root.glob(SESSION_DB_GLOB))
        if not database_paths:
            raise DataContractError(f"session metadata.sqlite3が見つからない: {self.data_root}")

        samples: list[ReadySample] = []
        issues: list[str] = []
        for database_path in database_paths:
            found, problems = self._load_database(database_path)
            samples.extend(found)
            issues.extend(problems)
        return SampleLoadResult(tuple(samples), tuple(issues))

    def validate_sample(self, sample: ReadySample) -> None:
        """sampleが最初の承認時点と同じ状態を保っているか確認する。

        Args:
            sample: 読込時点の承認情報とfile hashを保持するsample。

        Raises:
            DataContractError: DB上の承認状態または元ファイルが変化した場合。
            OSError: 入力fileを読み取れない場合。
        """
        with self._connection(sample.database_path) as connection:
            row = connection.execute(REVALIDATE_QUERY, sample.row_key).fetchone()
        if row is None or _fingerprint(dict(row)) != sample.fingerprint:
            raise DataContractError(f"SAM3の承認・採用状態が変更または失効した: {sample.sample_id}")
        if sha256_of_file(sample.rgb_path) != sample.rgb_sha256:
            raise DataContractError(f"元RGBのhashが変更されている: {sample.sample_id}")
        if sha256_of_file(sample.mask_path) != sample.mask_sha256:
            raise DataContractError(f"元maskのhashが変更されている: {sample.sample_id}")
        if not sample.depth_path.is_file():
            raise DataContractError(f"CNN受け渡し用depthが存在しない: {sample.sample_id}")

    def _load_database(self, database_path: Path) -> tuple[list[ReadySample], list[str]]:
        """1つのsession DBを走査し、sample生成または不正行の回収を行う。

        Args:
            database_path: 読み込むsession SQLiteのpath。

        Returns:
            検証済みsampleと、そのDBから収集したissue一覧。
        """
        capture_date = database_path.parents[1].name
        samples: list[ReadySample] = []
        issues: list[str] = []
        with self._connection(database_path) as connection:
            try:
                cursor = connection.execute(READY_QUERY)
            except sqlite3.Error as error:
                return [], [f"{database_path}: {READY_TABLE}を読めない: {error}"]
            columns = {column[0] for column in cursor.description}
            missing = REQUIRED_READY_COLUMNS - columns
            if missing:
                return [], [f"{database_path}: 必須列がない: {', '.join(sorted(missing))}"]
            for row in cursor.fetchall():
                try:
                    samples.append(self._to_sample(dict(row), capture_date, database_path))
                except (DataContractError, KeyError, TypeError, ValueError) as error:
                    issues.append(f"{database_path.parent.name}/{row['image_id']}: {error}")
        return samples, issues

    def _to_sample(self, row: dict[str, Any], capture_date: str, database_path: Path) -> ReadySample:
        """DB rowを検証し、共有root内の安全なfile参照へ変換する。

        Args:
            row: `sam3_dino_results`の1行。
            capture_date: rowが所属するcapture date。
            database_path: rowを取得したsession SQLiteのpath。

        Returns:
            DB状態と元ファイルを識別する情報を持つsample。

        Raises:
            DataContractError: 必須情報やasset pathが不正な場合。
            KeyError: 必須列がrowにない場合。
        """
        if not row["source_path"] or not row["mask_path"] or not row["depth_path"]:
            raise DataContractError("RGB・depth・maskのpathが揃っていない")
        if not row["approval_id"] or not row["review_event_id"]:
            raise DataContractError("画像承認または物体採用イベントがない")

        fingerprint = _fingerprint(row)
        return ReadySample(
            sample_id=f"{SAMPLE_ID_PREFIX}{fingerprint}",
            capture_date=capture_date,
            session_id=str(row["session_id"]),
            image_id=str(row["image_id"]),
            instrument_uid=str(row["instrument_uid"]),
            rgb_path=self._resolve_asset(str(row["source_path"])),
            mask_path=self._resolve_asset(str(row["mask_path"])),
            depth_path=self._resolve_asset(str(row["depth_path"])),
            rgb_sha256=str(row["source_sha256"]),
            mask_sha256=str(row["mask_sha256"]),
            database_path=database_path,
            row_key=(
                str(row["session_id"]),
                str(row["image_id"]),
                str(row["prediction_id"]),
                str(row["revision"]),
                int(row["object_index"]),
            ),
            fingerprint=fingerprint,
        )

    def _resolve_asset(self, value: str) -> Path:
        """共有root内にある相対asset pathを安全に解決する。

        Args:
            value: DBに保存されたPOSIX形式の相対path。

        Returns:
            存在が確認されたassetの絶対path。

        Raises:
            DataContractError: pathがroot外を指す、またはfileが存在しない場合。
        """
        relative = PurePosixPath(value)
        if not value or relative.is_absolute() or ".." in relative.parts or "\\" in value:
            raise DataContractError(f"asset pathが安全なroot相対pathではない: {value}")
        path = (self.data_root / Path(*relative.parts)).resolve()
        if not path.is_relative_to(self.data_root) or not path.is_file():
            raise DataContractError(f"assetが存在しないか共有root外を指している: {value}")
        return path

    @contextmanager
    def _connection(self, database_path: Path) -> Iterator[sqlite3.Connection]:
        """SQLiteへ読み取り専用で接続し、処理終了後に接続を閉じる。

        Args:
            database_path: 接続するSQLite file。

        Yields:
            row factoryを設定したSQLite connection。
        """
        connection = sqlite3.connect(f"{database_path.as_uri()}?mode=ro", timeout=self.timeout_seconds, uri=True)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()


def _fingerprint(row: dict[str, Any]) -> str:
    """DB row を安定した SHA-256 の識別子へ変換して、改ざんを検出する。"""
    canonical = json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
