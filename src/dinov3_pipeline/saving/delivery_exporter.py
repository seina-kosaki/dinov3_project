"""CNN bundleとdebug出力を公開する。"""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np

from dinov3_pipeline.common.constants import (
    CNN_DIR,
    DEBUG_DIR,
    DEPTH_DIR,
    DEPTH_FORMAT,
    DEPTH_SUFFIX,
    FEATURE_DIR,
    FEATURE_DTYPE,
    FEATURE_LAYOUT,
    FEATURE_NDIM,
    FEATURE_SUFFIX,
    IMAGE_FORMAT,
    IMAGE_SUFFIX,
    ISSUES_FILE,
    MASK_DIR,
    METADATA_DB,
    PCA_DIR,
    RGB_DIR,
    SAFE_NAME_PATTERN,
)
from dinov3_pipeline.common.models import DataContractError, DeliveryOutput, FeatureArtifact, ReadySample
from dinov3_pipeline.data_loading.preprocessing import prepare_crop
from dinov3_pipeline.saving.pca_visualizer import save_pca_visualization

SCHEMA = """
CREATE TABLE dataset_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE samples(
    sample_id TEXT PRIMARY KEY,
    instrument_uid TEXT NOT NULL,
    session_id TEXT NOT NULL,
    image_id TEXT NOT NULL,
    capture_date TEXT NOT NULL,
    rgb_path TEXT NOT NULL,
    depth_path TEXT NOT NULL,
    mask_path TEXT NOT NULL,
    feature_path TEXT NOT NULL,
    feature_block_1_path TEXT NOT NULL,
    feature_block_2_path TEXT NOT NULL,
    feature_block_3_path TEXT NOT NULL,
    feature_block_4_path TEXT NOT NULL
);
"""
SAMPLE_COLUMNS = 13
FEATURE_BLOCK_DIRS = tuple(f"feature_block_{index}" for index in range(1, 5))


@dataclass
class _WriteResult:
    rows: list[tuple[str, ...]] = field(default_factory=list)
    debug_dirs: set[PurePosixPath] = field(default_factory=set)
    issues: list[str] = field(default_factory=list)
    visualization_count: int = 0


def export_delivery(
    delivery_root: Path,
    crop_size: int,
    dataset_version: str,
    records: list[tuple[ReadySample, FeatureArtifact]],
    issues: list[str],
    mask_depth: bool = True,
) -> DeliveryOutput:
    if re.fullmatch(SAFE_NAME_PATTERN, dataset_version) is None:
        raise DataContractError(f"dataset versionが不正である: {dataset_version}")
    if not records:
        raise DataContractError("CNN bundleへ出力できる承認済みsampleがない")
    models = {(artifact.model_name, artifact.weights_sha256) for _, artifact in records}
    if len(models) != 1:
        raise DataContractError("同一bundle内でDINOv3モデルまたは重みを混在できない")

    root = Path(delivery_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / CNN_DIR / dataset_version
    if destination.exists():
        raise FileExistsError(f"CNN bundleが既に存在する: {destination}")

    staging = Path(tempfile.mkdtemp(prefix=f".{dataset_version}-", dir=root))
    try:
        bundle = staging / CNN_DIR / dataset_version
        debug = staging / DEBUG_DIR
        bundle.mkdir(parents=True, exist_ok=True)
        for name in (RGB_DIR, DEPTH_DIR, MASK_DIR, FEATURE_DIR, *FEATURE_BLOCK_DIRS):
            (bundle / name).mkdir(parents=True, exist_ok=True)

        result = _write_samples(records, crop_size, dataset_version, bundle, debug, mask_depth)
        if not result.rows:
            raise DataContractError("CNN bundleへ出力できるcropがない")
        model_name, _ = next(iter(models))
        _write_database(
            bundle / METADATA_DB,
            dataset_version,
            crop_size,
            model_name,
            result.rows,
            mask_depth,
        )
        all_issues = [*issues, *result.issues]
        _write_issues(debug / dataset_version / ISSUES_FILE, all_issues)
        _publish(staging, root, dataset_version, result.debug_dirs)
        return DeliveryOutput(destination, root / DEBUG_DIR, len(result.rows), result.visualization_count, len(all_issues))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _write_samples(
    records: list[tuple[ReadySample, FeatureArtifact]],
    crop_size: int,
    dataset_version: str,
    bundle: Path,
    debug: Path,
    mask_depth: bool,
) -> _WriteResult:
    result = _WriteResult()
    for sample, artifact in records:
        key = hashlib.sha256(sample.sample_id.encode("utf-8")).hexdigest()
        rgb = PurePosixPath(RGB_DIR) / f"{key}{IMAGE_SUFFIX}"
        depth = PurePosixPath(DEPTH_DIR) / f"{key}{DEPTH_SUFFIX}"
        mask = PurePosixPath(MASK_DIR) / f"{key}{IMAGE_SUFFIX}"
        feature = PurePosixPath(FEATURE_DIR) / f"{key}{FEATURE_SUFFIX}"
        feature_blocks = tuple(
            PurePosixPath(directory) / f"{key}{FEATURE_SUFFIX}"
            for directory in FEATURE_BLOCK_DIRS
        )
        try:
            _validate_segment(sample.capture_date, "capture date")
            _validate_segment(sample.session_id, "session ID")
            cropped = prepare_crop(sample, crop_size, mask_depth)
            if len(artifact.block_paths) != len(FEATURE_BLOCK_DIRS):
                raise DataContractError("DINO特徴量には4つのblock出力が必要である")
            feature_sources = (artifact.path, *artifact.block_paths)
            feature_destinations = (feature, *feature_blocks)
            for source in feature_sources:
                data = np.load(source, allow_pickle=False)
                if (
                    data.dtype != np.dtype(FEATURE_DTYPE)
                    or data.ndim != FEATURE_NDIM
                    or data.shape != artifact.shape
                ):
                    raise DataContractError(
                        f"DINO特徴量は{artifact.shape}の{FEATURE_DTYPE} {FEATURE_LAYOUT} NPYでなければならない"
                    )

            cropped.rgb.save(bundle / rgb, format=IMAGE_FORMAT)
            cropped.mask.save(bundle / mask, format=IMAGE_FORMAT)
            cropped.depth.save(bundle / depth, format=DEPTH_FORMAT)
            for source, destination in zip(feature_sources, feature_destinations):
                shutil.copy2(source, bundle / destination)

            debug_dir = PurePosixPath(dataset_version) / sample.capture_date / sample.session_id
            try:
                save_pca_visualization(
                    tuple(bundle / path for path in (*feature_blocks, feature)),
                    bundle / rgb,
                    bundle / mask,
                    debug / debug_dir / PCA_DIR / f"{key}{IMAGE_SUFFIX}",
                )
                result.visualization_count += 1
                result.debug_dirs.add(debug_dir)
            except DataContractError as error:
                result.issues.append(f"{sample.session_id}/{sample.image_id}: PCA可視化: {error}")

            result.rows.append(
                (
                    sample.sample_id,
                    sample.instrument_uid,
                    sample.session_id,
                    sample.image_id,
                    sample.capture_date,
                    rgb.as_posix(),
                    depth.as_posix(),
                    mask.as_posix(),
                    feature.as_posix(),
                    *(path.as_posix() for path in feature_blocks),
                )
            )
        except (DataContractError, OSError, ValueError) as error:
            result.issues.append(f"{sample.session_id}/{sample.image_id}: export: {error}")
    return result


def _write_database(
    path: Path,
    dataset_version: str,
    crop_size: int,
    model_name: str,
    rows: list[tuple[str, ...]],
    mask_depth: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCHEMA)
        connection.executemany(
            "INSERT INTO dataset_metadata(key, value) VALUES (?, ?)",
            (
                ("dataset_version", dataset_version),
                ("crop_size", str(crop_size)),
                ("mask_depth", str(mask_depth).lower()),
                ("feature_layout", FEATURE_LAYOUT),
                ("feature_dtype", FEATURE_DTYPE),
                ("model_name", model_name),
            ),
        )
        placeholders = ", ".join("?" * SAMPLE_COLUMNS)
        connection.executemany(f"INSERT INTO samples VALUES ({placeholders})", rows)
        connection.commit()
    finally:
        connection.close()


def _write_issues(path: Path, issues: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("reason",))
        writer.writerows((issue,) for issue in issues)


def _publish(staging: Path, root: Path, dataset_version: str, debug_dirs: set[PurePosixPath]) -> None:
    debug_root = root / DEBUG_DIR
    issues_relative = Path(dataset_version) / ISSUES_FILE
    moves = [(staging / DEBUG_DIR / rel, debug_root / rel) for rel in sorted(debug_dirs)]
    moves.append((staging / DEBUG_DIR / issues_relative, debug_root / issues_relative))
    for _, target in moves:
        if target.exists():
            raise FileExistsError(f"debug出力が既に存在する: {target}")

    cnn_target = root / CNN_DIR / dataset_version
    cnn_target.parent.mkdir(parents=True, exist_ok=True)
    (staging / CNN_DIR / dataset_version).replace(cnn_target)
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)


def _validate_segment(value: str, label: str) -> None:
    if re.fullmatch(SAFE_NAME_PATTERN, value) is None:
        raise DataContractError(f"{label}がdirectory名として不正: {value}")
