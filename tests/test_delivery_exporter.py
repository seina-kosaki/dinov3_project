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
from contextlib import closing
from dataclasses import dataclass, replace
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


@pytest.mark.parametrize("table", ["dataset_metadata", "samples"])
def test_export_delivery_schema_types_nullability_and_primary_keys(
    delivered: Delivered, table: str
) -> None:
    """出力値の必須性と、PRAGMAが示すDDL制約を区別して検証する。"""
    names = (
        ["key", "value"]
        if table == "dataset_metadata"
        else [
            "sample_id", "instrument_uid", "session_id", "image_id", "capture_date",
            "rgb_path", "depth_path", "mask_path", "feature_path",
        ]
    )
    with sqlite3.connect(delivered.output.cnn_path / "metadata.sqlite3") as connection:
        columns = list(connection.execute(f"PRAGMA table_info({table})"))
        rows = list(connection.execute(f"SELECT * FROM {table}"))  # noqa: S608 - tableは固定のテストパラメータ
        assert [(row[1], row[2], row[3], row[4], row[5]) for row in columns] == [
            (name, "TEXT", int(index != 0), None, int(index == 0))
            for index, name in enumerate(names)
        ]
        assert rows
        assert all(isinstance(value, str) for row in rows for value in row)
        # TEXT PRIMARY KEYだけではNULLを拒否しない（確認後はrollbackする）。
        connection.execute(
            f"INSERT INTO {table} VALUES ({', '.join('?' for _ in names)})",  # noqa: S608 - tableは固定のテストパラメータ
            (None, *("test" for _ in names[1:])),
        )
        connection.rollback()


def test_export_delivery_metadata_matches_contract(delivered: Delivered) -> None:
    """7件のmetadataをTEXTで保存する。"""
    with sqlite3.connect(delivered.output.cnn_path / "metadata.sqlite3") as connection:
        metadata = dict(connection.execute("SELECT key, value FROM dataset_metadata"))
    assert metadata == {
        "dataset_version": VERSION,
        "crop_size": str(CROP_SIZE),
        "mask_depth": "true",
        "feature_layout": "HWC",
        "feature_dtype": "float32",
        "model_name": delivered.artifact.model_name,
        "weights_sha256": delivered.artifact.weights_sha256,
    }


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


@pytest.mark.parametrize("mask_depth", [False, True])
def test_export_delivery_depth_background_matches_selected_mask_setting(
    make_sample: SampleFactory, tmp_path: Path, mask_depth: bool
) -> None:
    """depth背景とmetadataがmask_depthの選択に一致する。"""
    # 準備
    sample = make_sample(size=64, mask_box=(24, 24, 40, 40))
    raw_depth = np.full((64, 64), 2.5, dtype=np.float32)
    raw_depth[24:40, 24:40] = DEPTH_VALUE
    Image.fromarray(raw_depth, mode="F").save(sample.depth_path)
    artifact = _artifact(tmp_path)

    # 実行
    output = export_delivery(
        tmp_path / "delivery",
        CROP_SIZE,
        VERSION,
        [(sample, artifact)],
        [],
        mask_depth=mask_depth,
    )
    row = _row(Delivered(output, sample, artifact, tmp_path / "delivery"))

    # 検証
    with Image.open(output.cnn_path / row[6]) as depth:
        assert depth.mode == "F"
        assert depth.getpixel((0, 0)) == pytest.approx(0 if mask_depth else 2.5)
        assert depth.getpixel((CROP_SIZE // 2, CROP_SIZE // 2)) == pytest.approx(
            DEPTH_VALUE
        )
    with sqlite3.connect(output.cnn_path / "metadata.sqlite3") as connection:
        metadata = dict(
            connection.execute("SELECT key, value FROM dataset_metadata").fetchall()
        )
    assert metadata["mask_depth"] == str(mask_depth).lower()
    with Image.open(output.cnn_path / row[5]) as rgb:
        assert rgb.getpixel((0, 0)) == (0, 0, 0)


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


def test_export_delivery_oversized_sample_is_excluded_with_diagnostics(
    make_sample: SampleFactory,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """exportでのcrop超過はissueに一度記録し、他のsampleを公開する。"""
    valid = make_sample(mask_box=(16, 16, 48, 48))
    oversized_mask = valid.mask_path.with_name("oversized-mask.png")
    image = Image.new("L", (64, 64), 0)
    image.paste(255, (0, 0, 64, 16))
    image.save(oversized_mask)
    oversized = replace(valid, sample_id="oversized", mask_path=oversized_mask)
    artifact = _artifact(tmp_path)
    output = export_delivery(
        tmp_path / "delivery", CROP_SIZE, VERSION,
        [(oversized, artifact), (valid, artifact)], [],
    )
    assert output.exported_count == 1
    assert output.issue_count == 1
    issues = (output.debug_path / "runs" / VERSION / "issues.csv").read_text(encoding="utf-8-sig")
    assert issues.count("対象maskが設定crop_sizeに収まらない") == 1
    assert "oversized, crop_size=32, foreground_width=64, foreground_height=16" in issues
    assert capsys.readouterr() == ("", "")
    assert not caplog.records


@pytest.mark.parametrize("failure_at", ["none", "cnn", "debug", "issues"])
def test_export_delivery_publication_order_and_failure_state(
    make_sample: SampleFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_at: str,
) -> None:
    """完成CNNを先に公開し、debug失敗時も移動済み成果物は戻さない。"""
    sample = make_sample(mask_box=(16, 16, 48, 48))
    artifact = _artifact(tmp_path)
    root = tmp_path / "delivery"
    cnn = root / "cnn" / VERSION
    debug = root / "debug" / sample.capture_date / sample.session_id / VERSION
    issues = root / "debug" / "runs" / VERSION / "issues.csv"
    original_replace = Path.replace
    observed = []

    def observe_replace(source: Path, target: Path) -> Path:
        """移動時点でDBと全参照ファイルが読み取れることも確認する。"""
        phase = "cnn" if target == cnn else "issues" if target == issues else "debug"
        observed.append(phase)
        if phase == "cnn":
            assert not cnn.exists()
            with closing(sqlite3.connect(source / "metadata.sqlite3")) as connection:
                row = connection.execute("SELECT * FROM samples").fetchone()
                assert all((source / relative).is_file() for relative in row[5:])
                assert connection.execute("SELECT COUNT(*) FROM dataset_metadata").fetchone() == (7,)
        else:
            assert (cnn / "metadata.sqlite3").is_file()
        if phase == failure_at:
            raise OSError("publish failed")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", observe_replace)
    if failure_at == "none":
        export_delivery(root, CROP_SIZE, VERSION, [(sample, artifact)], [])
    else:
        with pytest.raises(OSError, match="publish failed"):
            export_delivery(root, CROP_SIZE, VERSION, [(sample, artifact)], [])
    expected_order = ["cnn", "debug", "issues"]
    assert observed == (
        expected_order if failure_at == "none"
        else expected_order[:expected_order.index(failure_at) + 1]
    )
    assert cnn.exists() == (failure_at != "cnn")
    assert debug.exists() == (failure_at in {"none", "issues"})
    assert issues.exists() == (failure_at == "none")
    assert not list(root.glob(f".{VERSION}-*"))


def test_export_delivery_existing_debug_prevents_cnn_publication(
    make_sample: SampleFactory, tmp_path: Path
) -> None:
    """debugの衝突はCNNの移動前に検出し、既存ファイルを維持する。"""
    sample = make_sample()
    root = tmp_path / "delivery"
    issues = root / "debug" / "runs" / VERSION / "issues.csv"
    issues.parent.mkdir(parents=True)
    issues.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="debug出力"):
        export_delivery(root, CROP_SIZE, VERSION, [(sample, _artifact(tmp_path))], [])
    assert not (root / "cnn" / VERSION).exists()
    assert issues.read_text(encoding="utf-8") == "existing"
    assert not list(root.glob(f".{VERSION}-*"))
