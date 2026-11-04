"""CNN bundleとdebug出力の公開仕様を確認する。

責務:
    export_delivery が、crop・特徴量・最小SQLite・日付/session階層のPCA画像を出力することを確認する。

処理の流れ:
    小さな特徴量とcropを使って export_delivery を実行し、出力の構成と内容を検証する。

関連ファイル:
    src/dinov3_pipeline/infrastructure/delivery_exporter.py、tests/conftest.py。

設計意図:
    CNNチームが受け取るbundleの構成と最小schemaを、変更に気づけるよう固定する。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from conftest import DEPTH_VALUE, SampleFactory
from PIL import Image

from dinov3_pipeline.domain.models import (
    DataContractError,
    DeliveryOutput,
    FeatureArtifact,
    ReadySample,
)
from dinov3_pipeline.infrastructure.delivery_exporter import export_delivery

CROP_SIZE = 32
VERSION = "capture_v001"
SAMPLE_COLUMNS = {
    "sample_id",
    "instrument_uid",
    "session_id",
    "image_id",
    "capture_date",
    "rgb_path",
    "depth_path",
    "mask_path",
    "feature_path",
}


@dataclass(frozen=True)
class Delivered:
    """出力結果と、その元になった入力をまとめる。"""

    output: DeliveryOutput
    sample: ReadySample
    artifact: FeatureArtifact
    root: Path


def _artifact(tmp_path: Path, weights_sha256: str = "weights-1") -> FeatureArtifact:
    """小さなHWC特徴量を保存し、artifactを返す。

    Args:
        tmp_path: 保存先。
        weights_sha256: 重みのSHA-256。

    Returns:
        保存済みの特徴量 artifact。
    """
    path = tmp_path / f"feature-{weights_sha256}.npy"
    np.save(path, np.arange(16, dtype=np.float32).reshape(2, 2, 4), allow_pickle=False)
    return FeatureArtifact(path, (2, 2, 4), "float32", "dinov3_test", weights_sha256)


@pytest.fixture
def delivered(make_sample: SampleFactory, tmp_path: Path) -> Delivered:
    """1物体を出力し、結果と入力を返す。

    Args:
        make_sample: 承認済み物体の生成関数。
        tmp_path: 一時ディレクトリ。

    Returns:
        出力結果と入力。
    """
    sample = make_sample(size=64, mask_box=(16, 16, 48, 48))
    artifact = _artifact(tmp_path)
    root = tmp_path / "delivery"
    output = export_delivery(root, CROP_SIZE, VERSION, [(sample, artifact)], [])
    return Delivered(output, sample, artifact, root)


def _row(delivered: Delivered) -> tuple[str, ...]:
    """出力したSQLiteの先頭行を返す。"""
    with sqlite3.connect(delivered.output.cnn_path / "metadata.sqlite3") as connection:
        row: tuple[str, ...] = connection.execute("SELECT * FROM samples").fetchone()
    return row


@pytest.mark.parametrize("directory", ["rgb", "depth", "masks", "features"])
def test_export_delivery_valid_sample_creates_asset_directory(
    delivered: Delivered, directory: str
) -> None:
    """CNN bundleに、各アセットのディレクトリを作る。"""
    # 準備・実行
    bundle = delivered.output.cnn_path

    # 検証
    assert any((bundle / directory).iterdir())


def test_export_delivery_valid_sample_writes_only_metadata_and_samples_tables(
    delivered: Delivered,
) -> None:
    """CNN用SQLiteのテーブルを2つに絞る。"""
    # 準備・実行
    with sqlite3.connect(delivered.output.cnn_path / "metadata.sqlite3") as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    # 検証
    assert tables == {"dataset_metadata", "samples"}


def test_export_delivery_valid_sample_writes_minimal_sample_columns(
    delivered: Delivered,
) -> None:
    """samplesの列を、識別子とパスだけに絞る。"""
    # 準備・実行
    with sqlite3.connect(delivered.output.cnn_path / "metadata.sqlite3") as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(samples)")}

    # 検証
    assert columns == SAMPLE_COLUMNS


def test_export_delivery_valid_sample_records_instrument_session_and_image_ids(
    delivered: Delivered,
) -> None:
    """器具ID・session ID・image IDを記録する。"""
    # 準備・実行
    row = _row(delivered)

    # 検証
    assert row[1:5] == ("INS-000001", "session-001", "image-001", "2026-10-30")


def test_export_delivery_valid_sample_keeps_float_depth_crop(
    delivered: Delivered,
) -> None:
    """depth cropがfloat値を保つ。"""
    # 準備・実行
    depth_path = delivered.output.cnn_path / _row(delivered)[6]

    # 検証
    with Image.open(depth_path) as depth:
        assert depth.mode == "F"
        assert depth.getpixel((CROP_SIZE // 2, CROP_SIZE // 2)) == pytest.approx(
            DEPTH_VALUE
        )


def test_export_delivery_valid_sample_copies_feature_npy_unchanged(
    delivered: Delivered,
) -> None:
    """特徴量NPYを内容を変えずにbundleへコピーする。"""
    # 準備・実行
    copied = np.load(delivered.output.cnn_path / _row(delivered)[8], allow_pickle=False)

    # 検証
    np.testing.assert_array_equal(
        copied, np.load(delivered.artifact.path, allow_pickle=False)
    )


def test_export_delivery_valid_sample_writes_pca_under_date_and_session(
    delivered: Delivered,
) -> None:
    """PCA画像を debug/<日付>/<session>/<version>/pca_rgb に保存する。"""
    # 準備・実行
    pca_dir = (
        delivered.output.debug_path / "2026-10-30" / "session-001" / VERSION / "pca_rgb"
    )

    # 検証
    assert len(list(pca_dir.glob("*.png"))) == 1


def test_export_delivery_valid_sample_writes_issues_csv(delivered: Delivered) -> None:
    """除外理由のCSVを debug/runs に保存する。"""
    # 準備・実行
    issues = delivered.output.debug_path / "runs" / VERSION / "issues.csv"

    # 検証
    assert issues.is_file()


def test_export_delivery_existing_version_raises_file_exists_error(
    delivered: Delivered,
) -> None:
    """同じversionへの再出力は上書きせず拒否する。"""
    # 準備・実行・検証
    with pytest.raises(FileExistsError):
        export_delivery(
            delivered.root,
            CROP_SIZE,
            VERSION,
            [(delivered.sample, delivered.artifact)],
            [],
        )


def test_export_delivery_mixed_weights_raises_contract_error(
    make_sample: SampleFactory, tmp_path: Path
) -> None:
    """異なる重みの特徴量を1つのbundleに混ぜない。"""
    # 準備
    sample = make_sample(size=64, mask_box=(16, 16, 48, 48))
    records = [
        (sample, _artifact(tmp_path, "weights-1")),
        (sample, _artifact(tmp_path, "weights-2")),
    ]

    # 実行・検証
    with pytest.raises(DataContractError, match="混在"):
        export_delivery(tmp_path / "delivery", CROP_SIZE, VERSION, records, [])


@pytest.mark.parametrize("version", ["../escape", "", "bad version"])
def test_export_delivery_unsafe_version_raises_contract_error(
    make_sample: SampleFactory, tmp_path: Path, version: str
) -> None:
    """ディレクトリ名として安全でないversionを拒否する。"""
    # 準備
    sample = make_sample(size=64, mask_box=(16, 16, 48, 48))

    # 実行・検証
    with pytest.raises(DataContractError, match="versionが不正"):
        export_delivery(
            tmp_path / "delivery",
            CROP_SIZE,
            version,
            [(sample, _artifact(tmp_path))],
            [],
        )
