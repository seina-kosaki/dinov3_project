from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from dinov3_pipeline.saving.dataset_version import reserve_dataset_version


def test_empty_root_and_repeated_reservations(tmp_path: Path) -> None:
    root = tmp_path / "delivery"
    assert reserve_dataset_version(root) == "capture_v001"
    assert reserve_dataset_version(root) == "capture_v002"
    assert not (root / "cnn").exists()
    assert not (root / "debug").exists()


@pytest.mark.parametrize(
    "relative",
    [
        "cnn/capture_v007",
        "debug/capture_v007/2026-10-30/session-001/pca_rgb",
        "debug/capture_v007",
        "debug/runs/capture_v007",
        "debug/2026-10-30/session-001/capture_v007",
        ".dataset_versions/capture_v007",
    ],
)
def test_uses_maximum_across_outputs_and_reservations(tmp_path: Path, relative: str) -> None:
    existing = tmp_path / relative
    existing.mkdir(parents=True)
    sentinel = existing / "keep.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    (tmp_path / "cnn" / "capture_v002").mkdir(parents=True, exist_ok=True)
    assert reserve_dataset_version(tmp_path) == "capture_v008"
    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def test_ignores_unrelated_names_and_grows_past_three_digits(tmp_path: Path) -> None:
    for name in ("capture_v999", "capture_v1000-extra", "custom", "capture_vbad"):
        (tmp_path / "cnn" / name).mkdir(parents=True)
    assert reserve_dataset_version(tmp_path) == "capture_v1000"


def test_numbered_file_is_not_overwritten(tmp_path: Path) -> None:
    (tmp_path / "cnn").mkdir()
    existing = tmp_path / "cnn" / "capture_v003"
    existing.write_text("keep", encoding="utf-8")
    assert reserve_dataset_version(tmp_path) == "capture_v004"
    assert existing.read_text(encoding="utf-8") == "keep"


def test_explicit_version_is_reserved_and_never_reused(tmp_path: Path) -> None:
    assert reserve_dataset_version(tmp_path, "legacy") == "legacy"
    with pytest.raises(FileExistsError):
        reserve_dataset_version(tmp_path, "legacy")
    assert reserve_dataset_version(tmp_path, "capture_v012") == "capture_v012"
    assert reserve_dataset_version(tmp_path) == "capture_v013"


@pytest.mark.parametrize(
    "relative",
    ["cnn", "debug", "debug/runs", "debug/2026-10-30/session-001"],
)
def test_explicit_version_rejects_any_existing_output(tmp_path: Path, relative: str) -> None:
    (tmp_path / relative / "legacy").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        reserve_dataset_version(tmp_path, "legacy")


@pytest.mark.parametrize("remaining", ["pca", "issues", "file"])
@pytest.mark.parametrize("version", ["capture_v007", "legacy"])
def test_new_debug_only_rejects_explicit_version(tmp_path: Path, remaining: str, version: str) -> None:
    existing = tmp_path / "debug" / version
    if remaining == "pca":
        existing = existing / "2026-10-30" / "session-001" / "pca_rgb" / "sample.png"
    elif remaining == "issues":
        existing = existing / "issues.csv"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"unchanged")
    with pytest.raises(FileExistsError):
        reserve_dataset_version(tmp_path, version)
    assert reserve_dataset_version(tmp_path) == ("capture_v008" if version == "capture_v007" else "capture_v001")
    assert existing.read_bytes() == b"unchanged"


def test_new_debug_date_and_session_are_not_versions(tmp_path: Path) -> None:
    existing = tmp_path / "debug" / "legacy" / "2026-10-30" / "capture_v999" / "pca_rgb"
    existing.mkdir(parents=True)
    assert reserve_dataset_version(tmp_path) == "capture_v001"
    assert reserve_dataset_version(tmp_path, "capture_v999") == "capture_v999"
    assert reserve_dataset_version(tmp_path, "2026-10-30") == "2026-10-30"


@pytest.mark.parametrize("relative", ["debug/runs/capture_v007", "debug/2026-10-30/session-001/capture_v007"])
def test_legacy_numbered_file_is_protected(tmp_path: Path, relative: str) -> None:
    existing = tmp_path / relative
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"unchanged")
    assert reserve_dataset_version(tmp_path) == "capture_v008"
    with pytest.raises(FileExistsError):
        reserve_dataset_version(tmp_path, "capture_v007")
    assert existing.read_bytes() == b"unchanged"


@pytest.mark.parametrize("version", ["../escape", "", "/absolute", "a/b"])
def test_invalid_explicit_version_does_not_create_root(tmp_path: Path, version: str) -> None:
    root = tmp_path / "delivery"
    with pytest.raises(ValueError):
        reserve_dataset_version(root, version)
    assert not root.exists()


def test_parallel_auto_reservations_are_unique(tmp_path: Path) -> None:
    barrier = Barrier(8)

    def reserve() -> str:
        barrier.wait(timeout=10)
        return reserve_dataset_version(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        versions = list(executor.map(lambda _: reserve(), range(8)))
    assert set(versions) == {f"capture_v{number:03d}" for number in range(1, 9)}


def test_parallel_explicit_reservations_have_one_winner(tmp_path: Path) -> None:
    barrier = Barrier(4)

    def reserve() -> bool:
        barrier.wait(timeout=10)
        try:
            reserve_dataset_version(tmp_path, "legacy")
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: reserve(), range(4)))
    assert results.count(True) == 1


def test_parallel_process_reservations_are_unique(tmp_path: Path) -> None:
    with ProcessPoolExecutor(max_workers=4) as executor:
        versions = list(executor.map(reserve_dataset_version, [tmp_path] * 12))
    assert set(versions) == {f"capture_v{number:03d}" for number in range(1, 13)}
