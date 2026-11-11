"""ファイル内容のSHA-256を計算する。"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_of_file(path: Path) -> str:
    """対象ファイルのSHA-256を返す。"""
    path = Path(path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
