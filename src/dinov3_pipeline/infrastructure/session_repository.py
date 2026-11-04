"""日付・sessionごとの撮影SQLiteから、DINO承認済みの物体を読み取る。

責務:
    shared_data 配下の全session DBを読み取り専用で走査し、is_dino_ready=1 の行だけを
    ReadySample に変換する。抽出前後に承認状態と原本hashを再検証する。
    画像処理やCNN向け出力は担わない。

処理の流れ:
    1. <日付>/<session_id>/metadata.sqlite3 を列挙する。
    2. 各DBから ready 行を読み、パスの安全性と必須列を確認して ReadySample にする。
    3. validate_sample で、行の指紋・RGB/mask hash・depth存在を再確認する。

関連ファイル:
    constants.py、domain/models.py、usecase/delivery.py、entrypoints/deliver.py。

設計意図:
    撮影側DBは各sessionの正本であるため書き込まず、DINO用テーブルも追加しない。
    承認取消・原本改変を抽出の前後で検出し、古い承認で出力しないようにする。

Note:
    SQLiteは mode=ro で開く。承認状態はこのクラスの外で変更されうるため、結果をキャッシュしない。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from dinov3_pipeline.constants import (
    READY_TABLE,
    REQUIRED_READY_COLUMNS,
    SAMPLE_ID_PREFIX,
    SESSION_DB_GLOB,
)
from dinov3_pipeline.domain.models import (
    DataContractError,
    ReadySample,
    SampleLoadResult,
)
from dinov3_pipeline.infrastructure.file_hash import sha256_of_file

READY_QUERY = (
    f"SELECT * FROM {READY_TABLE} WHERE is_dino_ready = 1 "
    "ORDER BY session_id, image_id, revision, object_index"
)
REVALIDATE_QUERY = (
    f"SELECT * FROM {READY_TABLE} WHERE session_id = ? AND image_id = ? "
    "AND prediction_id = ? AND revision = ? AND object_index = ? AND is_dino_ready = 1"
)


class SessionSam3Repository:
    """session DBを変更せず、readyフラグが立った物体だけを取得する。

    Args:
        data_root: ``<日付>/<session_id>/metadata.sqlite3`` を含む共有root。
        timeout_seconds: SQLite読み取りのlock待機秒数。
    """

    def __init__(self, data_root: Path, timeout_seconds: float) -> None:
        self.data_root = Path(data_root).resolve()
        self.timeout_seconds = timeout_seconds

    def load_ready_samples(self) -> SampleLoadResult:
        """全session DBからDINO承認済みの物体を読み込む。

        Returns:
            読み込めた物体と、除外した行・DBの理由。

        Raises:
            DataContractError: 共有rootにsession DBが1つも無い場合。
        """
        database_paths = sorted(self.data_root.glob(SESSION_DB_GLOB))
        if not database_paths:
            raise DataContractError(
                f"session metadata.sqlite3が見つからない: {self.data_root}"
            )

        samples: list[ReadySample] = []
        issues: list[str] = []
        for database_path in database_paths:
            # 1つのDBが壊れていても他のsessionを処理できるよう、理由を集めて続行する
            found, problems = self._load_database(database_path)
            samples.extend(found)
            issues.extend(problems)
        return SampleLoadResult(tuple(samples), tuple(issues))

    def validate_sample(self, sample: ReadySample) -> None:
        """承認状態と原本が、読込時点から変わっていないことを確認する。

        Args:
            sample: 読込済みの承認済み物体。

        Raises:
            DataContractError: 承認の取消・変更、原本の改変、depthの欠落がある場合。
        """
        with self._connection(sample.database_path) as connection:
            row = connection.execute(REVALIDATE_QUERY, sample.row_key).fetchone()
        # 行全体の指紋で比較し、承認者・採用イベントなどの変更も検出する
        if row is None or _fingerprint(dict(row)) != sample.fingerprint:
            raise DataContractError(
                f"SAM3の承認・採用状態が変更または失効した: {sample.sample_id}"
            )
        if sha256_of_file(sample.rgb_path) != sample.rgb_sha256:
            raise DataContractError(f"元RGBのhashが変更されている: {sample.sample_id}")
        if sha256_of_file(sample.mask_path) != sample.mask_sha256:
            raise DataContractError(f"元maskのhashが変更されている: {sample.sample_id}")
        if not sample.depth_path.is_file():
            raise DataContractError(
                f"CNN受け渡し用depthが存在しない: {sample.sample_id}"
            )

    def _load_database(
        self, database_path: Path
    ) -> tuple[list[ReadySample], list[str]]:
        """1つのsession DBから承認済み物体を読み込む。

        Args:
            database_path: session DBのパス。

        Returns:
            読み込めた物体と、除外理由。
        """
        capture_date = database_path.parents[1].name
        samples: list[ReadySample] = []
        issues: list[str] = []
        with self._connection(database_path) as connection:
            try:
                cursor = connection.execute(READY_QUERY)
            except sqlite3.Error as error:
                return [], [f"{database_path}: {READY_TABLE}を読めない: {error}"]
            # 古いschemaのDBは列が足りないため、行を読む前に契約を確認する
            columns = {column[0] for column in cursor.description}
            missing = REQUIRED_READY_COLUMNS - columns
            if missing:
                return [], [
                    f"{database_path}: 必須列がない: {', '.join(sorted(missing))}"
                ]
            for row in cursor.fetchall():
                try:
                    samples.append(
                        self._to_sample(dict(row), capture_date, database_path)
                    )
                except (DataContractError, KeyError, TypeError, ValueError) as error:
                    issues.append(
                        f"{database_path.parent.name}/{row['image_id']}: {error}"
                    )
        return samples, issues

    def _to_sample(
        self, row: dict[str, Any], capture_date: str, database_path: Path
    ) -> ReadySample:
        """SQL行を、rootの内側に固定した承認済み物体へ変換する。

        Args:
            row: ready行。
            capture_date: 撮影日。
            database_path: 行を読んだDB。

        Returns:
            承認済み物体。

        Raises:
            DataContractError: パスが欠落・不正、または承認情報が無い場合。
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
        """DB由来のroot相対パスを検証し、共有root内の絶対パスへ解決する。

        Args:
            value: DBに記録されたroot相対パス。

        Returns:
            存在するファイルの絶対パス。

        Raises:
            DataContractError: 絶対パス・親参照・root外・不在の場合。
        """
        relative = PurePosixPath(value)
        # DBの値でroot外のファイルを読ませないよう、相対パスのみ許可する
        if (
            not value
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in value
        ):
            raise DataContractError(f"asset pathが安全なroot相対pathではない: {value}")
        path = (self.data_root / Path(*relative.parts)).resolve()
        if not path.is_relative_to(self.data_root) or not path.is_file():
            raise DataContractError(
                f"assetが存在しないか共有root外を指している: {value}"
            )
        return path

    @contextmanager
    def _connection(self, database_path: Path) -> Iterator[sqlite3.Connection]:
        """書き込みを許可せず、利用後に必ず閉じるSQLite接続を開く。

        Args:
            database_path: 開くDB。

        Yields:
            読み取り専用の接続。
        """
        connection = sqlite3.connect(
            f"{database_path.as_uri()}?mode=ro",
            timeout=self.timeout_seconds,
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()


def _fingerprint(row: dict[str, Any]) -> str:
    """行全体から再現可能な指紋（SHA-256）を作る。

    Args:
        row: DB行の辞書。

    Returns:
        16進のSHA-256。
    """
    canonical = json.dumps(
        row, sort_keys=True, ensure_ascii=True, allow_nan=False, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
