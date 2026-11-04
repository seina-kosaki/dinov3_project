"""ファイル内容のハッシュを計算する。

責務:
    ファイルのSHA-256を返す。それ以外は担わない。

処理の流れ:
    ファイルをバイナリで開き、標準ライブラリでSHA-256を計算する。

関連ファイル:
    infrastructure/session_repository.py（原本の改変検出）、infrastructure/dinov3_extractor.py（重みの識別）。

設計意図:
    原本と重みの同一性確認で同じ処理が必要になるため、1か所に置く。
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_of_file(path: Path) -> str:
    """ファイル内容のSHA-256を返す。

    Args:
        path: 対象ファイル。

    Returns:
        16進のSHA-256。
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
