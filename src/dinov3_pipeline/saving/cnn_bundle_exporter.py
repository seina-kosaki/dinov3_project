"""CNN bundleとdebug出力を公開する。"""

from __future__ import annotations

import csv
import hashlib
import json
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
from dinov3_pipeline.common.models import DataContractError, DeliveryOutput, FeatureSet, ReadySample
from dinov3_pipeline.data_loading.preprocessing import CroppedInputs
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
    original_rgb_path TEXT NOT NULL,
    original_mask_path TEXT NOT NULL,
    original_depth_path TEXT NOT NULL,
    crop_box_json TEXT NOT NULL,
    feature_path TEXT NOT NULL,
    feature_block_1_path TEXT NOT NULL,
    feature_block_2_path TEXT NOT NULL,
    feature_block_3_path TEXT NOT NULL,
    feature_block_4_path TEXT NOT NULL
);
"""
FEATURE_BLOCK_DIRS = tuple(f"feature_block_{index}" for index in range(1, 5))
ORIGINAL_RGB_DIR = "original_rgb"
ORIGINAL_MASK_DIR = "original_masks"
ORIGINAL_DEPTH_DIR = "original_depth"
FEATURE_KEYS = (
    "feature",
    "feature_block_1",
    "feature_block_2",
    "feature_block_3",
    "feature_block_4",
)
SAMPLE_INSERT_QUERY = (
    "INSERT INTO samples VALUES "
    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


@dataclass
class _WriteResult:
    """1回の export で作った sample 行・debug 出力・問題一覧をまとめて保持する。"""

    rows: list[tuple[str, ...]] = field(default_factory=list)
    debug_dirs: set[PurePosixPath] = field(default_factory=set)
    issues: list[str] = field(default_factory=list)
    visualization_count: int = 0


def export_cnn_bundle(
    delivery_root: Path,
    crop_size: int,
    dataset_version: str,
    records: list[tuple[ReadySample, CroppedInputs, FeatureSet]],
    issues: list[str],
    # 学習メモ: `*`より後ろは名前付きで渡し、位置引数の取り違えを防ぐ。
    *,
    mask_depth: bool = True,
    mask_rgb_background: bool = True,
) -> DeliveryOutput:
    """承認済みsampleをCNN bundleとdebug出力へ書き出す。

    Args:
        delivery_root: 成果物を公開するroot directory。
        crop_size: 保存するcropの1辺の長さ。
        dataset_version: bundleのversion名。
        records: sample、整形済みmodalities、特徴量の組。
        issues: 入力処理中に記録されたissue一覧。
        mask_depth: Trueなら保存depthのmask外を0にした設定値。
        mask_rgb_background: PCA可視化で前景だけを使う設定値。

    Returns:
        CNN bundleとdebug出力の生成結果。

    Raises:
        DataContractError: version、record、モデル構成が契約に合わない場合。
        FileExistsError: 同じversionの出力が既に存在する場合。
    """
    if re.fullmatch(SAFE_NAME_PATTERN, dataset_version) is None:
        raise DataContractError(f"dataset versionが不正である: {dataset_version}")
    if not records:
        raise DataContractError("CNN bundleへ出力できる承認済みsampleがない")
    model_names = {feature_set.model_name for _, _, feature_set in records}
    if len(model_names) != 1:
        raise DataContractError("同一bundle内でDINOv3モデルを混在できない")

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
        for name in (
            RGB_DIR,
            DEPTH_DIR,
            MASK_DIR,
            ORIGINAL_RGB_DIR,
            ORIGINAL_MASK_DIR,
            ORIGINAL_DEPTH_DIR,
            FEATURE_DIR,
            *FEATURE_BLOCK_DIRS,
        ):
            (bundle / name).mkdir(parents=True, exist_ok=True)

        result = _write_samples(
            records,
            crop_size,
            dataset_version,
            bundle,
            debug,
            mask_rgb_background=mask_rgb_background,
        )
        if not result.rows:
            raise DataContractError("CNN bundleへ出力できるcropがない")
        model_name = next(iter(model_names))
        _write_database(
            bundle / METADATA_DB,
            dataset_version,
            crop_size,
            model_name,
            result.rows,
            mask_depth=mask_depth,
        )
        all_issues = [*issues, *result.issues]
        _write_issues(debug / dataset_version / ISSUES_FILE, all_issues)
        _publish(staging, root, dataset_version, result.debug_dirs)
        return DeliveryOutput(destination, root / DEBUG_DIR, len(result.rows), result.visualization_count, len(all_issues))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _write_samples(
    records: list[tuple[ReadySample, CroppedInputs, FeatureSet]],
    crop_size: int,
    dataset_version: str,
    bundle: Path,
    debug: Path,
    *,
    mask_rgb_background: bool,
) -> _WriteResult:
    """整形済みsampleと特徴量を保存し、SQLite行とdebug出力を準備する。

    Args:
        records: sample、整形済みmodalities、特徴量の組。
        crop_size: 整形済み画像の期待辺長。
        dataset_version: debug出力のdirectory名。
        bundle: 一時bundle directory。
        debug: 一時debug directory。
        mask_rgb_background: PCA可視化で前景だけを使う設定値。

    Returns:
        保存成功行、debug directory、issueをまとめた結果。
    """
    result = _WriteResult()
    for sample, cropped, feature_set in records:
        key = hashlib.sha256(sample.sample_id.encode("utf-8")).hexdigest()
        rgb = PurePosixPath(RGB_DIR) / f"{key}{IMAGE_SUFFIX}"
        depth = PurePosixPath(DEPTH_DIR) / f"{key}{DEPTH_SUFFIX}"
        mask = PurePosixPath(MASK_DIR) / f"{key}{IMAGE_SUFFIX}"
        original_rgb = PurePosixPath(ORIGINAL_RGB_DIR) / f"{key}{sample.rgb_path.suffix}"
        original_mask = PurePosixPath(ORIGINAL_MASK_DIR) / f"{key}{sample.mask_path.suffix}"
        original_depth = PurePosixPath(ORIGINAL_DEPTH_DIR) / f"{key}{sample.depth_path.suffix}"
        feature = PurePosixPath(FEATURE_DIR) / f"{key}{FEATURE_SUFFIX}"
        feature_blocks = tuple(
            PurePosixPath(directory) / f"{key}{FEATURE_SUFFIX}"
            for directory in FEATURE_BLOCK_DIRS
        )
        try:
            _validate_segment(sample.capture_date, "capture date")
            _validate_segment(sample.session_id, "session ID")
            _validate_cropped_inputs(cropped, crop_size)
            _validate_feature_set(feature_set.features)
            _copy_original_image(sample.rgb_path, bundle / original_rgb)
            _copy_original_image(sample.mask_path, bundle / original_mask)
            _copy_original_image(sample.depth_path, bundle / original_depth)
            feature_paths = {
                "feature": feature,
                "feature_block_1": feature_blocks[0],
                "feature_block_2": feature_blocks[1],
                "feature_block_3": feature_blocks[2],
                "feature_block_4": feature_blocks[3],
            }
            for key_name, relative_path in feature_paths.items():
                np.save(
                    bundle / relative_path,
                    feature_set.features[key_name],
                    allow_pickle=False,
                )

            cropped.rgb.save(bundle / rgb, format=IMAGE_FORMAT)
            cropped.mask.save(bundle / mask, format=IMAGE_FORMAT)
            cropped.depth.save(bundle / depth, format=DEPTH_FORMAT)

            debug_dir = PurePosixPath(dataset_version) / sample.capture_date / sample.session_id
            try:
                save_pca_visualization(
                    tuple(bundle / path for path in (*feature_blocks, feature)),
                    bundle / rgb,
                    bundle / mask,
                    debug / debug_dir / PCA_DIR / f"{key}{IMAGE_SUFFIX}",
                    mask_rgb_background,
                )
                result.visualization_count += 1
                result.debug_dirs.add(debug_dir)
            except DataContractError as error:
                result.issues.append(f"{sample.session_id}/{sample.image_id}: PCA可視化: {error}")

            x1, y1, x2, y2 = cropped.crop_box
            crop_box_json = json.dumps(
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                separators=(",", ":"),
            )
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
                    original_rgb.as_posix(),
                    original_mask.as_posix(),
                    original_depth.as_posix(),
                    crop_box_json,
                    feature.as_posix(),
                    *(path.as_posix() for path in feature_blocks),
                )
            )
        except (DataContractError, OSError, ValueError) as error:
            result.issues.append(f"{sample.session_id}/{sample.image_id}: export: {error}")
    return result


def _validate_cropped_inputs(cropped: CroppedInputs, crop_size: int) -> None:
    """RGB・mask・depthが設定crop_sizeの同じ寸法であることを確認する。"""
    expected_size = (crop_size, crop_size)
    if cropped.rgb.size != expected_size:
        raise DataContractError("整形済みRGB・mask・depthは設定crop_sizeの同じ寸法である必要がある")
    if cropped.mask.size != expected_size or cropped.depth.size != expected_size:
        raise DataContractError("整形済みRGB・mask・depthは設定crop_sizeの同じ寸法である必要がある")
    x1, y1, x2, y2 = cropped.crop_box
    has_expected_width = x2 - x1 == crop_size
    has_expected_height = y2 - y1 == crop_size
    has_nonnegative_origin = x1 >= 0 and y1 >= 0
    if not has_expected_width or not has_expected_height or not has_nonnegative_origin:
        raise DataContractError("crop_boxは元画像上の有効なhalf-open crop範囲である必要がある")


def _copy_original_image(source: Path, destination: Path) -> None:
    """入力画像ファイルを再エンコードせずbundleへコピーする。"""
    shutil.copyfile(source, destination)


def _validate_feature_set(features: dict[str, np.ndarray]) -> None:
    """5種類の特徴量key、dtype、shapeがbundle契約を満たすことを確認する。"""
    if set(features) != set(FEATURE_KEYS):
        raise DataContractError("DINO特徴量にはfeatureと4つのblock出力が必要である")

    expected_shape: tuple[int, ...] | None = None
    for key_name in FEATURE_KEYS:
        feature = features[key_name]
        if (
            feature.dtype != np.dtype(FEATURE_DTYPE)
            or feature.ndim != FEATURE_NDIM
            or feature.shape[0] <= 0
            or feature.shape[1] <= 0
            or feature.shape[2] <= 0
        ):
            raise DataContractError(
                f"DINO特徴量は{FEATURE_DTYPE} {FEATURE_LAYOUT} NPYでなければならない"
            )
        if expected_shape is not None and feature.shape != expected_shape:
            raise DataContractError("5種類のDINO特徴量のshapeが一致しない")
        expected_shape = feature.shape


def _write_database(
    path: Path,
    dataset_version: str,
    crop_size: int,
    model_name: str,
    rows: list[tuple[str, ...]],
    *,
    mask_depth: bool,
) -> None:
    """dataset_metadataとsamplesをSQLiteへ書き出す。

    Args:
        path: 作成するSQLite database path。
        dataset_version: 出力bundleのversion。
        crop_size: RGB・depth・maskのcrop辺長。
        model_name: 特徴量生成に使用したモデルID。
        rows: samples tableへ挿入する行。
        mask_depth: 保存時のdepth背景mask設定。
    """
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
        connection.executemany(SAMPLE_INSERT_QUERY, rows)
        connection.commit()
    finally:
        connection.close()


def _write_issues(path: Path, issues: list[str]) -> None:
    """debug 配下に出力した問題の一覧を CSV として保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("reason",))
        writer.writerows((issue,) for issue in issues)


def _publish(staging: Path, root: Path, dataset_version: str, debug_dirs: set[PurePosixPath]) -> None:
    """staging から最終的な delivery_root へ成果物を移動する。"""
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
    """session_id や capture_date などの安全な path segment かどうか検証する。"""
    if re.fullmatch(SAFE_NAME_PATTERN, value) is None:
        raise DataContractError(f"{label}がdirectory名として不正: {value}")
