"""既存成果物と予約記録から、再利用しないdelivery versionを確保する。

予約は原子的なディレクトリ作成で行い、失敗した実行の番号も再利用しない。
同じ予約方式を使わない外部の書き込みは排他対象としない。
"""

from __future__ import annotations

import re
from pathlib import Path

from dinov3_pipeline.constants import CNN_DIR, DEBUG_DIR, RUNS_DIR, SAFE_NAME_PATTERN

RESERVATIONS_DIR = ".dataset_versions"
AUTO_VERSION_PATTERN = re.compile(r"capture_v([0-9]+)")


def _output_versions(root: Path) -> set[str]:
    """CNN・実行issue・session別debugのいずれかに残るversion名を集める。"""
    versions = {path.name for path in (root / CNN_DIR).glob("*")}
    versions.update(path.name for path in (root / DEBUG_DIR / RUNS_DIR).glob("*"))
    versions.update(path.name for path in (root / DEBUG_DIR).glob("*") if path.name != RUNS_DIR)
    versions.update(path.name for path in (root / DEBUG_DIR).glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/*/*"))
    return versions


def reserve_dataset_version(delivery_root: Path, dataset_version: str | None = None) -> str:
    """未使用のversionを予約し、その名前を返す。

    Args:
        delivery_root: 既存成果物と予約記録を確認する送付root。
        dataset_version: 明示名。Noneなら既存・予約済み番号の最大値に1を加える。

    Returns:
        予約済みのversion名。自動採番はcapture_v001から始まり、最低3桁となる。

    Raises:
        ValueError: 明示名が安全なディレクトリ名でない場合。
        FileExistsError: 明示名の成果物または予約記録が既にある場合。
        OSError: 探索または予約ディレクトリの作成に失敗した場合。

    Note:
        予約記録は成功・失敗を問わず残す。予約済み番号は欠番でも再利用しない。
        自動採番の予約競合時は既存番号を読み直し、別の番号を予約する。
    """
    if dataset_version is not None and re.fullmatch(SAFE_NAME_PATTERN, dataset_version) is None:
        raise ValueError("dataset_versionは安全なディレクトリ名で指定する必要がある")

    root = Path(delivery_root).resolve()
    reservations = root / RESERVATIONS_DIR
    reservations.mkdir(parents=True, exist_ok=True)
    while True:
        used = _output_versions(root)
        used.update(path.name for path in reservations.iterdir())
        if dataset_version is None:
            numbers = [int(match.group(1)) for name in used if (match := AUTO_VERSION_PATTERN.fullmatch(name)) is not None]
            version = f"capture_v{max(numbers, default=0) + 1:03d}"
        else:
            version = dataset_version
            if version in used:
                raise FileExistsError(f"dataset versionは使用済みである: {version}")
        try:
            (reservations / version).mkdir()
        except FileExistsError:
            if dataset_version is not None:
                raise
            continue
        if version in _output_versions(root):
            if dataset_version is not None:
                raise FileExistsError(f"dataset versionは使用済みである: {version}")
            continue
        return version
