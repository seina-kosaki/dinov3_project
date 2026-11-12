from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from dinov3_pipeline.common.models import FeatureArtifact
from dinov3_pipeline.entrypoints import deliver
from dinov3_pipeline.saving import delivery_exporter
from dinov3_pipeline.settings import load_settings


class ArrayExtractor:
    def __init__(self, *args: object) -> None:
        pass

    def extract(self, sample: object, destination: Path) -> FeatureArtifact:
        np.save(destination, np.arange(16, dtype=np.float32).reshape(2, 2, 4))
        block_paths = []
        for index in range(1, 5):
            block_path = destination.parent / f"feature_block_{index}" / destination.name
            block_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(block_path, np.full((2, 2, 4), index, dtype=np.float32))
            block_paths.append(block_path)
        return FeatureArtifact(
            destination,
            (2, 2, 4),
            "float32",
            "dinov3_test",
            "weights-1",
            tuple(block_paths),
        )


@pytest.fixture
def delivery_root(
    session_root: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    root = tmp_path / "delivery"
    settings = load_settings(
        data_root=session_root[0],
        delivery_root=root,
        crop_size=64,
        dataset_version=None,
    )
    monkeypatch.setattr(deliver, "load_settings", lambda: settings)
    monkeypatch.setattr(deliver, "Dinov3FeatureExtractor", ArrayExtractor)
    return root


def _assert_version_outputs(root: Path, version: str) -> None:
    with sqlite3.connect(root / "cnn" / version / "metadata.sqlite3") as connection:
        metadata = dict(connection.execute("SELECT key, value FROM dataset_metadata"))
    assert metadata["dataset_version"] == version
    assert metadata["mask_depth"] == "false"
    assert (root / "debug" / version / "issues.csv").is_file()
    assert len(list((root / "debug" / version).glob("*/*/pca_rgb/*.png"))) == 1


def test_main_repeated_runs_use_same_version_everywhere_without_overwriting(delivery_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=deliver.__name__)
    assert deliver.main() == 0
    first_files = {path: path.read_bytes() for path in delivery_root.rglob("*") if path.is_file()}
    assert deliver.main() == 0
    for version in ("capture_v001", "capture_v002"):
        _assert_version_outputs(delivery_root, version)
        assert f"delivery {version}:" in caplog.text
    assert all(path.read_bytes() == data for path, data in first_files.items())


def test_main_model_failure_consumes_version_without_publishing(delivery_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object) -> None:
        raise RuntimeError("model failed")

    monkeypatch.setattr(deliver, "Dinov3FeatureExtractor", fail)
    assert deliver.main() == 2
    assert (delivery_root / ".dataset_versions" / "capture_v001").is_dir()
    assert not (delivery_root / "cnn").exists()
    monkeypatch.setattr(deliver, "Dinov3FeatureExtractor", ArrayExtractor)
    assert deliver.main() == 0
    _assert_version_outputs(delivery_root, "capture_v002")


def test_main_export_failure_never_publishes_incomplete_cnn(delivery_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object) -> None:
        raise RuntimeError("database write failed")

    with monkeypatch.context() as patch:
        patch.setattr(delivery_exporter, "_write_database", fail)
        assert deliver.main() == 2
    assert not (delivery_root / "cnn").exists()
    assert not (delivery_root / "debug").exists()
    assert not list(delivery_root.glob(".capture_v001-*"))
    assert deliver.main() == 0
    _assert_version_outputs(delivery_root, "capture_v002")


def test_main_no_inputs_does_not_reserve_or_load_model(
    delivery_root: Path,
    session_root: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with sqlite3.connect(session_root[1]) as connection:
        connection.execute("UPDATE sam3_dino_results SET is_dino_ready = 0")

    def unexpected(*args: object) -> None:
        pytest.fail("入力なしでモデルを読み込んだ")

    monkeypatch.setattr(deliver, "Dinov3FeatureExtractor", unexpected)
    assert deliver.main() == 2
    assert not delivery_root.exists()


def test_main_explicit_version_is_preserved_and_duplicate_rejected_before_model(delivery_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = deliver.load_settings().model_copy(update={"dataset_version": "legacy"})
    monkeypatch.setattr(deliver, "load_settings", lambda: settings)
    assert deliver.main() == 0
    _assert_version_outputs(delivery_root, "legacy")

    def unexpected(*args: object) -> None:
        pytest.fail("使用済みversionでモデルを読み込んだ")

    monkeypatch.setattr(deliver, "Dinov3FeatureExtractor", unexpected)
    assert deliver.main() == 2
    _assert_version_outputs(delivery_root, "legacy")


@pytest.mark.parametrize(
    "relative",
    [
        "debug/{version}/2026-10-30/session-001/pca_rgb/sample.png",
        "debug/{version}/issues.csv",
        "debug/2026-10-30/session-001/{version}/pca_rgb/sample.png",
        "debug/runs/{version}/issues.csv",
    ],
)
@pytest.mark.parametrize("explicit", [False, True])
def test_main_protects_debug_only_versions(
    delivery_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
    explicit: bool,
) -> None:
    version = "legacy" if explicit else "capture_v007"
    existing = delivery_root / relative.format(version=version)
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"unchanged")
    if explicit:
        settings = deliver.load_settings().model_copy(update={"dataset_version": version})
        monkeypatch.setattr(deliver, "load_settings", lambda: settings)

        def unexpected(*args: object) -> None:
            pytest.fail("使用済みdebug versionでモデルを読み込んだ")

        monkeypatch.setattr(deliver, "Dinov3FeatureExtractor", unexpected)
        assert deliver.main() == 2
        assert not (delivery_root / "cnn").exists()
    else:
        assert deliver.main() == 0
        _assert_version_outputs(delivery_root, "capture_v008")
    assert existing.read_bytes() == b"unchanged"
