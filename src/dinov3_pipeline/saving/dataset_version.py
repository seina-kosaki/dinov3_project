"""dataset versionの予約と自動採番を行う。"""

from __future__ import annotations

import re
from pathlib import Path

from dinov3_pipeline.common.constants import CNN_DIR, DEBUG_DIR, RUNS_DIR, SAFE_NAME_PATTERN

RESERVATIONS_DIR = ".dataset_versions"
AUTO_VERSION_PATTERN = re.compile(r"capture_v([0-9]+)")


def _output_versions(root: Path) -> set[str]:
    versions = {path.name for path in (root / CNN_DIR).glob("*")}
    versions.update(path.name for path in (root / DEBUG_DIR / RUNS_DIR).glob("*"))
    versions.update(path.name for path in (root / DEBUG_DIR).glob("*") if path.name != RUNS_DIR)
    versions.update(path.name for path in (root / DEBUG_DIR).glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/*/*"))
    return versions


def reserve_dataset_version(delivery_root: Path, dataset_version: str | None = None) -> str:
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
