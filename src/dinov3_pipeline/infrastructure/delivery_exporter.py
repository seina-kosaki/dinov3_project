"""CNN用bundleとdebug可視化を、送付ディレクトリへ一括して公開する。

責務:
    承認済み物体ごとにRGB・depth・maskのcropとHWC特徴量を書き出し、最小SQLiteを作る。
    PCA可視化は日付/session階層のdebugへ分け、除外理由をissues.csvに記録する。
    特徴量の抽出や承認状態の検証は担わない。

処理の流れ:
    1. version・モデルの一意性・出力先の未存在を確認する。
    2. 一時ディレクトリへ、物体ごとのcrop・NPY・PCA画像を書き出す。
    3. 最小SQLiteとissues.csvを作り、完成後にcnn/ と debug/ へ移して公開する。

関連ファイル:
    infrastructure/preprocessing.py、infrastructure/pca_visualizer.py、usecase/delivery.py。

設計意図:
    CNNチームがbundleをコピーするだけで使えるよう、CNN用とdebug用を分け、
    途中失敗した出力が公開側に残らないよう一時ディレクトリ経由で公開する。

Note:
    同じdataset_versionへの再出力は上書きせずエラーにする。
"""

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

from dinov3_pipeline.constants import (
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
from dinov3_pipeline.domain.models import (
    DataContractError,
    DeliveryOutput,
    FeatureArtifact,
    ReadySample,
)
from dinov3_pipeline.infrastructure.pca_visualizer import save_pca_visualization
from dinov3_pipeline.infrastructure.preprocessing import prepare_crop

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
    feature_path TEXT NOT NULL
);
"""
SAMPLE_COLUMNS = 9


@dataclass
class _WriteResult:
    """物体ごとの書き出し結果（行・debug出力先・除外理由・PCA件数）を集める。"""

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
    """CNN bundleとdebug出力を、一時ディレクトリ経由で公開する。

    Args:
        delivery_root: ``cnn/`` と ``debug/`` を作る送付root。
        crop_size: RGB・depth・maskに共通のcrop辺長。
        dataset_version: 出力を識別する不変のversion。
        records: 承認済み物体と抽出済み特徴量の組。
        issues: 読込・抽出の段階で除外した理由。
        mask_depth: Trueならdepthのmask外を0にし、Falseならcrop内の生値を保存する。

    Returns:
        公開したbundleとdebugの場所、件数。

    Raises:
        DataContractError: versionが不正、モデルが混在、または出力できる物体が無い場合。
        FileExistsError: 同じversionの出力が既にある場合。
    """
    if re.fullmatch(SAFE_NAME_PATTERN, dataset_version) is None:
        raise DataContractError(f"dataset versionが不正である: {dataset_version}")
    if not records:
        raise DataContractError("CNN bundleへ出力できる承認済みsampleがない")
    # 特徴量の意味はモデルで変わるため、1つのbundleに異なるモデルを混ぜない
    models = {(artifact.model_name, artifact.weights_sha256) for _, artifact in records}
    if len(models) != 1:
        raise DataContractError("同一bundle内でDINOv3モデルまたは重みを混在できない")

    root = Path(delivery_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / CNN_DIR / dataset_version
    if destination.exists():
        raise FileExistsError(f"CNN bundleが既に存在する: {destination}")

    # 公開先と同じ場所に一時ディレクトリを作り、完成後の移動を同一ボリューム内で完結させる
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset_version}-", dir=root))
    try:
        bundle = staging / CNN_DIR / dataset_version
        debug = staging / DEBUG_DIR
        for name in (RGB_DIR, DEPTH_DIR, MASK_DIR, FEATURE_DIR):
            (bundle / name).mkdir(parents=True)

        result = _write_samples(records, crop_size, dataset_version, bundle, debug, mask_depth)
        if not result.rows:
            raise DataContractError("CNN bundleへ出力できるcropがない")
        model_name, weights_sha256 = next(iter(models))
        _write_database(
            bundle / METADATA_DB,
            dataset_version,
            crop_size,
            (model_name, weights_sha256),
            result.rows,
            mask_depth,
        )
        all_issues = [*issues, *result.issues]
        _write_issues(debug / dataset_version / ISSUES_FILE, all_issues)
        _publish(staging, root, dataset_version, result.debug_dirs)
        return DeliveryOutput(
            destination,
            root / DEBUG_DIR,
            len(result.rows),
            result.visualization_count,
            len(all_issues),
        )
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
    """物体ごとにcrop・特徴量・PCA画像を書き出す。

    Args:
        records: 承認済み物体と特徴量の組。
        crop_size: crop辺長。
        dataset_version: 出力version。
        bundle: 一時のCNN bundle。
        debug: 一時のdebug root。
        mask_depth: Trueならdepthのmask外を0にする。

    Returns:
        書き出せた行と、除外理由。
    """
    result = _WriteResult()
    for sample, artifact in records:
        key = hashlib.sha256(sample.sample_id.encode("utf-8")).hexdigest()
        rgb = PurePosixPath(RGB_DIR) / f"{key}{IMAGE_SUFFIX}"
        depth = PurePosixPath(DEPTH_DIR) / f"{key}{DEPTH_SUFFIX}"
        mask = PurePosixPath(MASK_DIR) / f"{key}{IMAGE_SUFFIX}"
        feature = PurePosixPath(FEATURE_DIR) / f"{key}{FEATURE_SUFFIX}"
        try:
            _validate_segment(sample.capture_date, "capture date")
            _validate_segment(sample.session_id, "session ID")
            cropped = prepare_crop(sample, crop_size, mask_depth)
            data = np.load(artifact.path, allow_pickle=False)
            if data.dtype != np.dtype(FEATURE_DTYPE) or data.ndim != FEATURE_NDIM:
                raise DataContractError(f"DINO特徴量は{FEATURE_DTYPE}の{FEATURE_LAYOUT} NPYでなければならない")

            cropped.rgb.save(bundle / rgb, format=IMAGE_FORMAT)
            cropped.mask.save(bundle / mask, format=IMAGE_FORMAT)
            cropped.depth.save(bundle / depth, format=DEPTH_FORMAT)
            shutil.copy2(artifact.path, bundle / feature)

            # PCAだけが失敗してもCNN向けデータは有効なため、理由を残して続行する
            debug_dir = PurePosixPath(dataset_version) / sample.capture_date / sample.session_id
            try:
                save_pca_visualization(
                    bundle / feature,
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
                )
            )
        except (DataContractError, OSError, ValueError) as error:
            # 1物体の失敗で他の物体の出力を止めないよう、理由を残して続行する
            result.issues.append(f"{sample.session_id}/{sample.image_id}: export: {error}")
    return result


def _write_database(
    path: Path,
    dataset_version: str,
    crop_size: int,
    model: tuple[str, str],
    rows: list[tuple[str, ...]],
    mask_depth: bool,
) -> None:
    """CNNが必要とするIDとbundle相対パスだけをSQLiteへ保存する。

    Args:
        path: SQLiteの保存先。
        dataset_version: 出力version。
        crop_size: crop辺長。
        model: （モデル名, 重みのSHA-256）。
        rows: samplesテーブルの行。
        mask_depth: depthにmaskを適用したか。
    """
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCHEMA)
        # 全物体に共通する再現情報は行ごとに繰り返さず、metadataへ1回だけ置く
        connection.executemany(
            "INSERT INTO dataset_metadata(key, value) VALUES (?, ?)",
            (
                ("dataset_version", dataset_version),
                ("crop_size", str(crop_size)),
                ("mask_depth", str(mask_depth).lower()),
                ("feature_layout", FEATURE_LAYOUT),
                ("feature_dtype", FEATURE_DTYPE),
                ("model_name", model[0]),
                ("weights_sha256", model[1]),
            ),
        )
        placeholders = ", ".join("?" * SAMPLE_COLUMNS)
        connection.executemany(f"INSERT INTO samples VALUES ({placeholders})", rows)
        connection.commit()
    finally:
        connection.close()


def _write_issues(path: Path, issues: list[str]) -> None:
    """除外理由をCSVに記録する。

    Args:
        path: CSVの保存先。
        issues: 除外理由。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Excelで文字化けしないよう、BOM付きUTF-8で書く
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("reason",))
        writer.writerows((issue,) for issue in issues)


def _publish(staging: Path, root: Path, dataset_version: str, debug_dirs: set[PurePosixPath]) -> None:
    """完成した一時ディレクトリを、公開先へ移動する。

    Args:
        staging: 一時ディレクトリ。
        root: 送付root。
        dataset_version: 出力version。
        debug_dirs: debug root配下の、公開するversionディレクトリ（相対）。

    Raises:
        FileExistsError: debugの出力先が既に存在する場合。
    """
    debug_root = root / DEBUG_DIR
    issues_relative = Path(dataset_version) / ISSUES_FILE
    moves = [(staging / DEBUG_DIR / rel, debug_root / rel) for rel in sorted(debug_dirs)]
    moves.append((staging / DEBUG_DIR / issues_relative, debug_root / issues_relative))
    # 既存のdebug成果物を上書きしないよう、CNN bundleを動かす前に確認する
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
    """DB由来の値が、出力root外へ逸脱するディレクトリ名でないことを確認する。

    Args:
        value: ディレクトリ名に使う値。
        label: エラー表示用の名称。

    Raises:
        DataContractError: 安全な名前でない場合。
    """
    if re.fullmatch(SAFE_NAME_PATTERN, value) is None:
        raise DataContractError(f"{label}がdirectory名として不正: {value}")
