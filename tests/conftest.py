"""テスト共通のフィクスチャ（承認済み物体・session DBの生成）を提供する。

責務:
    一時ディレクトリにRGB・mask・depthの実ファイルと、撮影側形式のsession DBを作る。
    個別の検証は担わない。

処理の流れ:
    各テストがフィクスチャ経由で ReadySample またはsession DBを受け取る。

関連ファイル:
    tests/ 配下の各テスト、src/dinov3_pipeline/domain/models.py。

設計意図:
    入力生成の重複を避け、テスト本体を「準備・実行・検証」だけに保つ。
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dinov3_pipeline.common.models import ReadySample

MaskBox = tuple[int, int, int, int]
SampleFactory = Callable[..., ReadySample]
DEPTH_VALUE = 1.25
RGB_COLOR = (120, 80, 40)


def _write_images(
    directory: Path, size: int, mask_box: MaskBox
) -> tuple[Path, Path, Path]:
    """物体領域だけに値を持つRGB・mask・depthを書き出す。

    Args:
        directory: 保存先。
        size: 正方形画像の辺長。
        mask_box: 物体領域（左, 上, 右, 下）。

    Returns:
        RGB・mask・depthのパス。
    """
    left, top, right, bottom = mask_box
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[top:bottom, left:right] = RGB_COLOR
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    depth = np.zeros((size, size), dtype=np.float32)
    depth[top:bottom, left:right] = DEPTH_VALUE
    directory.mkdir(parents=True, exist_ok=True)
    paths = (directory / "rgb.png", directory / "mask.png", directory / "depth.tiff")
    Image.fromarray(rgb).save(paths[0])
    Image.fromarray(mask).save(paths[1])
    Image.fromarray(depth, mode="F").save(paths[2])
    return paths


@pytest.fixture
def make_sample(tmp_path: Path) -> SampleFactory:
    """指定した大きさの承認済み物体（実ファイル付き）を作る関数を返す。

    Args:
        tmp_path: 一時ディレクトリ。

    Returns:
        ``make_sample(size=64, mask_box=(40, 36, 48, 44))`` で ReadySample を返す関数。
    """

    def factory(size: int = 64, mask_box: MaskBox = (40, 36, 48, 44)) -> ReadySample:
        """画像を書き出し、DBを参照しない ReadySample を返す。"""
        rgb, mask, depth = _write_images(tmp_path / "assets", size, mask_box)
        return ReadySample(
            sample_id="sample-001",
            capture_date="2026-10-30",
            session_id="session-001",
            image_id="image-001",
            instrument_uid="INS-000001",
            rgb_path=rgb,
            mask_path=mask,
            depth_path=depth,
            rgb_sha256="",
            mask_sha256="",
            database_path=tmp_path / "unused.sqlite3",
            row_key=("session-001", "image-001", "prediction-001", "revision-001", 0),
            fingerprint="fingerprint",
        )

    return factory


@pytest.fixture
def session_root(tmp_path: Path) -> tuple[Path, Path]:
    """承認済み1行と未承認1行を持つsession DBを、共有root内に作る。

    Args:
        tmp_path: 一時ディレクトリ。

    Returns:
        共有rootと、session DBのパス。
    """
    root = tmp_path / "shared_data"
    session_dir = root / "2026-10-30" / "session-001"
    rgb, mask, depth = _write_images(session_dir / "assets", 64, (0, 0, 64, 64))
    database_path = session_dir / "metadata.sqlite3"
    ready_values = (
        "session-001",
        "image-001",
        "prediction-001",
        "revision-001",
        "approval-001",
        0,
        1,
        "INS-000001",
        rgb.relative_to(root).as_posix(),
        hashlib.sha256(rgb.read_bytes()).hexdigest(),
        depth.relative_to(root).as_posix(),
        mask.relative_to(root).as_posix(),
        hashlib.sha256(mask.read_bytes()).hexdigest(),
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """CREATE TABLE sam3_dino_results(
                session_id TEXT, image_id TEXT, prediction_id TEXT, revision TEXT,
                approval_id TEXT, object_index INTEGER, review_event_id INTEGER,
                instrument_uid TEXT, source_path TEXT, source_sha256 TEXT,
                depth_path TEXT, mask_path TEXT, mask_sha256 TEXT, is_dino_ready INTEGER)"""
        )
        connection.execute(
            "INSERT INTO sam3_dino_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
            ready_values,
        )
        # 未承認行は image_id だけ変え、ready 行との区別をフラグのみに絞る
        unready_values = (ready_values[0], "unapproved-image", *ready_values[2:])
        connection.execute(
            "INSERT INTO sam3_dino_results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0)",
            unready_values,
        )
    return root, database_path
