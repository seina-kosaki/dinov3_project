"""CNN bundleとdebug出力の公開仕様を確認する。

責務:
    export_cnn_bundle が、crop・特徴量・最小SQLite・日付/session階層のPCA画像を出力することを確認する。

処理の流れ:
    小さな特徴量とcropを使って export_cnn_bundle を実行し、出力の構成と内容を検証する。

関連ファイル:
    src/dinov3_pipeline/infrastructure/cnn_bundle_exporter.py、tests/conftest.py。

設計意図:
    CNNチームが受け取るbundleの構成と最小schemaを、変更に気づけるよう固定する。
"""

from __future__ import annotations

import csv
import hashlib
import sqlite3
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from conftest import DEPTH_VALUE, SampleFactory
from PIL import Image

from dinov3_pipeline.common.models import (
    DataContractError,
    DeliveryOutput,
    FeatureArtifact,
    ReadySample,
)
from dinov3_pipeline.saving import cnn_bundle_exporter
from dinov3_pipeline.saving.cnn_bundle_exporter import export_cnn_bundle

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
    "feature_block_1_path",
    "feature_block_2_path",
    "feature_block_3_path",
    "feature_block_4_path",
}


@dataclass(frozen=True)
class Delivered:
    """出力結果と、その元になった入力をまとめる。"""

    output: DeliveryOutput
    sample: ReadySample
    artifact: FeatureArtifact
    root: Path


def _artifact(
    tmp_path: Path,
    file_suffix: str = "feature",
    model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
) -> FeatureArtifact:
    """小さなHWC特徴量を保存し、artifactを返す。

    Args:
        tmp_path: 保存先。
        file_suffix: 一時ファイル名の識別子。
        model_name: 特徴量を生成したモデルID。

    Returns:
        保存済みの特徴量 artifact。
    """
    path = tmp_path / f"{file_suffix}.npy"
    np.save(path, np.arange(16, dtype=np.float32).reshape(2, 2, 4), allow_pickle=False)
    block_paths = []
    for index in range(1, 5):
        block_path = tmp_path / f"feature_block_{index}-{file_suffix}.npy"
        np.save(
            block_path,
            np.full((2, 2, 4), index, dtype=np.float32),
            allow_pickle=False,
        )
        block_paths.append(block_path)
    return FeatureArtifact(
        path,
        (2, 2, 4),
        "float32",
        model_name,
        tuple(block_paths),
    )


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
    output = export_cnn_bundle(root, CROP_SIZE, VERSION, [(sample, artifact)], [])
    return Delivered(output, sample, artifact, root)


def _row(delivered: Delivered) -> tuple[str, ...]:
    """出力したSQLiteの先頭行を返す。"""
    with sqlite3.connect(delivered.output.cnn_path / "metadata.sqlite3") as connection:
        row: tuple[str, ...] = connection.execute("SELECT * FROM samples").fetchone()
    return row


@pytest.mark.parametrize(
    "directory",
    [
        "rgb",
        "depth",
        "masks",
        "features",
        "feature_block_1",
        "feature_block_2",
        "feature_block_3",
        "feature_block_4",
    ],
)
def test_export_cnn_bundle_valid_sample_creates_asset_directory(delivered: Delivered, directory: str) -> None:
    """CNN bundleに、各アセットのディレクトリを作る。"""
    # 準備・実行
    bundle = delivered.output.cnn_path

    # 検証
    assert any((bundle / directory).iterdir())


def test_export_cnn_bundle_valid_sample_writes_only_metadata_and_samples_tables(
    delivered: Delivered,
) -> None:
    """CNN用SQLiteのテーブルを2つに絞る。"""
    # 準備・実行
    with sqlite3.connect(delivered.output.cnn_path / "metadata.sqlite3") as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    # 検証
    assert tables == {"dataset_metadata", "samples"}


def test_export_cnn_bundle_valid_sample_writes_minimal_sample_columns(
    delivered: Delivered,
) -> None:
    """samplesの列を、識別子とパスだけに絞る。"""
    # 準備・実行
    with sqlite3.connect(delivered.output.cnn_path / "metadata.sqlite3") as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(samples)")}

    # 検証
    assert columns == SAMPLE_COLUMNS


@pytest.mark.parametrize("table", ["dataset_metadata", "samples"])
def test_export_cnn_bundle_schema_types_nullability_and_primary_keys(delivered: Delivered, table: str) -> None:
    """出力値の必須性と、PRAGMAが示すDDL制約を区別して検証する。"""
    names = (
        ["key", "value"]
        if table == "dataset_metadata"
        else [
            "sample_id",
            "instrument_uid",
            "session_id",
            "image_id",
            "capture_date",
            "rgb_path",
            "depth_path",
            "mask_path",
            "feature_path",
            "feature_block_1_path",
            "feature_block_2_path",
            "feature_block_3_path",
            "feature_block_4_path",
        ]
    )
    with sqlite3.connect(delivered.output.cnn_path / "metadata.sqlite3") as connection:
        columns = list(connection.execute(f"PRAGMA table_info({table})"))
        rows = list(connection.execute(f"SELECT * FROM {table}"))  # noqa: S608 - tableは固定のテストパラメータ
        assert [(row[1], row[2], row[3], row[4], row[5]) for row in columns] == [(name, "TEXT", int(index != 0), None, int(index == 0)) for index, name in enumerate(names)]
        assert rows
        assert all(isinstance(value, str) for row in rows for value in row)
        # TEXT PRIMARY KEYだけではNULLを拒否しない（確認後はrollbackする）。
        connection.execute(
            f"INSERT INTO {table} VALUES ({', '.join('?' for _ in names)})",  # noqa: S608 - tableは固定のテストパラメータ
            (None, *("test" for _ in names[1:])),
        )
        connection.rollback()


def test_export_cnn_bundle_metadata_matches_contract(delivered: Delivered) -> None:
    """6件のmetadataをTEXTで保存する。"""
    with sqlite3.connect(delivered.output.cnn_path / "metadata.sqlite3") as connection:
        metadata = dict(connection.execute("SELECT key, value FROM dataset_metadata"))
    assert metadata == {
        "dataset_version": VERSION,
        "crop_size": str(CROP_SIZE),
        "mask_depth": "true",
        "feature_layout": "HWC",
        "feature_dtype": "float32",
        "model_name": delivered.artifact.model_name,
    }


def test_export_cnn_bundle_valid_sample_records_instrument_session_and_image_ids(
    delivered: Delivered,
) -> None:
    """器具ID・session ID・image IDを記録する。"""
    # 準備・実行
    row = _row(delivered)

    # 検証
    assert row[1:5] == ("INS-000001", "session-001", "image-001", "2026-10-30")


def test_export_cnn_bundle_valid_sample_keeps_float_depth_crop(
    delivered: Delivered,
) -> None:
    """depth cropがfloat値を保つ。"""
    # 準備・実行
    depth_path = delivered.output.cnn_path / _row(delivered)[6]

    # 検証
    with Image.open(depth_path) as depth:
        assert depth.mode == "F"
        assert depth.getpixel((CROP_SIZE // 2, CROP_SIZE // 2)) == pytest.approx(DEPTH_VALUE)


@pytest.mark.parametrize("mask_depth", [False, True])
def test_export_cnn_bundle_depth_background_matches_selected_mask_setting(make_sample: SampleFactory, tmp_path: Path, mask_depth: bool) -> None:
    """depth背景とmetadataがmask_depthの選択に一致する。"""
    # 準備
    sample = make_sample(size=64, mask_box=(24, 24, 40, 40))
    raw_depth = np.full((64, 64), 2.5, dtype=np.float32)
    raw_depth[24:40, 24:40] = DEPTH_VALUE
    Image.fromarray(raw_depth, mode="F").save(sample.depth_path)
    artifact = _artifact(tmp_path)

    # 実行
    output = export_cnn_bundle(
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
        assert depth.getpixel((CROP_SIZE // 2, CROP_SIZE // 2)) == pytest.approx(DEPTH_VALUE)
    with sqlite3.connect(output.cnn_path / "metadata.sqlite3") as connection:
        metadata = dict(connection.execute("SELECT key, value FROM dataset_metadata").fetchall())
    assert metadata["mask_depth"] == str(mask_depth).lower()
    with Image.open(output.cnn_path / row[5]) as rgb:
        assert rgb.getpixel((0, 0)) == (0, 0, 0)


def test_export_cnn_bundle_without_rgb_mask_preserves_rgb_and_mask_output(
    make_sample: SampleFactory, tmp_path: Path
) -> None:
    """RGB背景除去を無効にしてもmask PNGを保存し、depth設定と分離する。"""
    sample = make_sample(size=64, mask_box=(24, 24, 40, 40))
    source_rgb = Image.new("RGB", (64, 64), (13, 14, 15))
    source_rgb.paste((120, 80, 40), (24, 24, 40, 40))
    source_rgb.save(sample.rgb_path)
    artifact = _artifact(tmp_path)
    output = export_cnn_bundle(
        tmp_path / "delivery",
        CROP_SIZE,
        VERSION,
        [(sample, artifact)],
        [],
        mask_depth=False,
        mask_rgb_background=False,
    )
    row = _row(Delivered(output, sample, artifact, tmp_path / "delivery"))

    with Image.open(output.cnn_path / row[5]) as rgb:
        assert rgb.size == (CROP_SIZE, CROP_SIZE)
        assert rgb.getpixel((0, 0)) == (13, 14, 15)
    with Image.open(output.cnn_path / row[7]) as mask:
        assert mask.mode == "L"
        assert mask.size == (CROP_SIZE, CROP_SIZE)
        assert mask.getpixel((0, 0)) == 0
    with Image.open(output.cnn_path / row[6]) as depth:
        assert depth.getpixel((0, 0)) == pytest.approx(0)
    with sqlite3.connect(output.cnn_path / "metadata.sqlite3") as connection:
        metadata = dict(connection.execute("SELECT key, value FROM dataset_metadata"))
    assert set(metadata) == {
        "dataset_version",
        "crop_size",
        "mask_depth",
        "feature_layout",
        "feature_dtype",
        "model_name",
    }


def test_export_cnn_bundle_valid_sample_copies_feature_npy_unchanged(
    delivered: Delivered,
) -> None:
    """最終特徴量と4つのblock NPYを内容を変えずにbundleへコピーする。"""
    row = _row(delivered)
    sources = (delivered.artifact.path, *delivered.artifact.block_paths)
    key = hashlib.sha256(delivered.sample.sample_id.encode("utf-8")).hexdigest()
    expected_paths = [
        f"features/{key}.npy",
        *(f"feature_block_{index}/{key}.npy" for index in range(1, 5)),
    ]
    assert list(row[8:]) == expected_paths
    for source, relative in zip(sources, row[8:]):
        copied = np.load(delivered.output.cnn_path / relative, allow_pickle=False)
        np.testing.assert_array_equal(copied, np.load(source, allow_pickle=False))
        assert copied.dtype == np.float32
        assert copied.shape == (2, 2, 4)


def test_export_cnn_bundle_valid_sample_writes_pca_under_date_and_session(
    delivered: Delivered,
) -> None:
    """5つのPCA-RGB画像とcrop RGBを debug/<version>/<日付>/<session>/pca_rgb に保存する。"""
    # 準備・実行
    pca_dir = delivered.output.debug_path / VERSION / "2026-10-30" / "session-001" / "pca_rgb"

    # 検証
    key = hashlib.sha256(delivered.sample.sample_id.encode("utf-8")).hexdigest()
    pca_path = pca_dir / f"{key}.png"
    assert [path.name for path in pca_dir.glob("*.png")] == [pca_path.name]
    with Image.open(pca_path) as pca:
        assert pca.size == (CROP_SIZE * 3, CROP_SIZE * 2)


def test_export_cnn_bundle_valid_sample_writes_issues_csv(delivered: Delivered) -> None:
    """除外理由のCSVを debug/<version> に保存する。"""
    # 準備・実行
    issues = delivered.output.debug_path / VERSION / "issues.csv"

    # 検証
    assert issues.is_file()


def test_export_cnn_bundle_existing_version_raises_file_exists_error(
    delivered: Delivered,
) -> None:
    """同じversionへの再出力は上書きせず拒否する。"""
    # 準備・実行・検証
    with pytest.raises(FileExistsError):
        export_cnn_bundle(
            delivered.root,
            CROP_SIZE,
            VERSION,
            [(delivered.sample, delivered.artifact)],
            [],
        )


def test_export_cnn_bundle_mixed_models_raises_contract_error(make_sample: SampleFactory, tmp_path: Path) -> None:
    """異なるモデルの特徴量を1つのbundleに混ぜない。"""
    # 準備
    sample = make_sample(size=64, mask_box=(16, 16, 48, 48))
    records = [
        (
            sample,
            _artifact(
                tmp_path,
                "model-s",
                "facebook/dinov3-vits16-pretrain-lvd1689m",
            ),
        ),
        (
            sample,
            _artifact(
                tmp_path,
                "model-b",
                "facebook/dinov3-vitb16-pretrain-lvd1689m",
            ),
        ),
    ]

    # 実行・検証
    with pytest.raises(DataContractError, match="混在"):
        export_cnn_bundle(tmp_path / "delivery", CROP_SIZE, VERSION, records, [])


@pytest.mark.parametrize("version", ["../escape", "", "bad version"])
def test_export_cnn_bundle_unsafe_version_raises_contract_error(make_sample: SampleFactory, tmp_path: Path, version: str) -> None:
    """ディレクトリ名として安全でないversionを拒否する。"""
    # 準備
    sample = make_sample(size=64, mask_box=(16, 16, 48, 48))

    # 実行・検証
    with pytest.raises(DataContractError, match="versionが不正"):
        export_cnn_bundle(
            tmp_path / "delivery",
            CROP_SIZE,
            version,
            [(sample, _artifact(tmp_path))],
            [],
        )


def test_export_cnn_bundle_oversized_sample_is_excluded_with_diagnostics(
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
    output = export_cnn_bundle(
        tmp_path / "delivery",
        CROP_SIZE,
        VERSION,
        [(oversized, artifact), (valid, artifact)],
        [],
    )
    assert output.exported_count == 1
    assert output.issue_count == 1
    issues = (output.debug_path / VERSION / "issues.csv").read_text(encoding="utf-8-sig")
    assert issues.count("対象maskが設定crop_sizeに収まらない") == 1
    assert "oversized, crop_size=32, foreground_width=64, foreground_height=16" in issues
    assert capsys.readouterr() == ("", "")
    assert not caplog.records


@pytest.mark.parametrize("failure_at", ["none", "cnn", "debug", "issues"])
def test_export_cnn_bundle_publication_order_and_failure_state(
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
    debug = root / "debug" / VERSION / sample.capture_date / sample.session_id
    issues = root / "debug" / VERSION / "issues.csv"
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
                assert connection.execute("SELECT COUNT(*) FROM dataset_metadata").fetchone() == (6,)
        else:
            assert (cnn / "metadata.sqlite3").is_file()
        if phase == failure_at:
            raise OSError("publish failed")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", observe_replace)
    if failure_at == "none":
        export_cnn_bundle(root, CROP_SIZE, VERSION, [(sample, artifact)], [])
    else:
        with pytest.raises(OSError, match="publish failed"):
            export_cnn_bundle(root, CROP_SIZE, VERSION, [(sample, artifact)], [])
    expected_order = ["cnn", "debug", "issues"]
    assert observed == (expected_order if failure_at == "none" else expected_order[: expected_order.index(failure_at) + 1])
    assert cnn.exists() == (failure_at != "cnn")
    assert debug.exists() == (failure_at in {"none", "issues"})
    assert issues.exists() == (failure_at == "none")
    assert not list(root.glob(f".{VERSION}-*"))


@pytest.mark.parametrize("relative", ["issues.csv", "2026-10-30/session-001/pca_rgb/keep.png"])
def test_export_cnn_bundle_existing_debug_prevents_cnn_publication(make_sample: SampleFactory, tmp_path: Path, relative: str) -> None:
    """debugの衝突はCNNの移動前に検出し、既存ファイルを維持する。"""
    sample = make_sample(mask_box=(16, 16, 48, 48))
    root = tmp_path / "delivery"
    issues = root / "debug" / VERSION / relative
    issues.parent.mkdir(parents=True)
    issues.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="debug出力"):
        export_cnn_bundle(root, CROP_SIZE, VERSION, [(sample, _artifact(tmp_path))], [])
    assert not (root / "cnn" / VERSION).exists()
    assert issues.read_text(encoding="utf-8") == "existing"
    assert not list(root.glob(f".{VERSION}-*"))


def test_export_cnn_bundle_pca_failure_keeps_cnn_and_records_issues(make_sample: SampleFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample = make_sample()
    artifact = _artifact(tmp_path)

    def fail(*args: object) -> None:
        raise DataContractError("PCA failed")

    monkeypatch.setattr(cnn_bundle_exporter, "save_pca_visualization", fail)
    output = export_cnn_bundle(tmp_path / "delivery", CROP_SIZE, VERSION, [(sample, artifact)], ["input issue"])
    assert output.exported_count == 1
    assert output.visualization_count == 0
    assert output.issue_count == 2
    with (output.debug_path / VERSION / "issues.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert list(csv.reader(stream)) == [["reason"], ["input issue"], ["session-001/image-001: PCA可視化: PCA failed"]]
    assert not list(output.debug_path.rglob("*.png"))
    copied = next((output.cnn_path / "features").glob("*.npy"))
    assert copied.read_bytes() == artifact.path.read_bytes()


def test_export_cnn_bundle_multiple_dates_and_sessions_publish_in_sorted_order(make_sample: SampleFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample = make_sample(mask_box=(16, 16, 48, 48))
    samples = [
        replace(sample, sample_id="third", capture_date="2026-10-31", session_id="session-002"),
        replace(sample, sample_id="second", session_id="session-002"),
        sample,
    ]
    artifact = _artifact(tmp_path)
    root = tmp_path / "delivery"
    legacy = root / "debug" / "2026-10-29" / "session-001" / "capture_v000" / "pca_rgb" / "keep.png"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"unchanged")
    original_replace = Path.replace
    observed = []

    def observe_replace(source: Path, target: Path) -> Path:
        observed.append(target)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", observe_replace)
    output = export_cnn_bundle(root, CROP_SIZE, VERSION, [(item, artifact) for item in samples], [])
    debug_dirs = sorted({root / "debug" / VERSION / item.capture_date / item.session_id for item in samples})
    assert observed == [output.cnn_path, *debug_dirs, root / "debug" / VERSION / "issues.csv"]
    assert output.exported_count == output.visualization_count == 3
    for item in samples:
        key = hashlib.sha256(item.sample_id.encode("utf-8")).hexdigest()
        pca = root / "debug" / VERSION / item.capture_date / item.session_id / "pca_rgb" / f"{key}.png"
        expected = tmp_path / f"expected-{key}.png"
        cnn_bundle_exporter.save_pca_visualization(
            tuple(
                output.cnn_path / f"feature_block_{index}" / f"{key}.npy"
                for index in range(1, 5)
            ) + (output.cnn_path / "features" / f"{key}.npy",),
            output.cnn_path / "rgb" / f"{key}.png",
            output.cnn_path / "masks" / f"{key}.png",
            expected,
        )
        assert pca.read_bytes() == expected.read_bytes()
    assert legacy.read_bytes() == b"unchanged"
    assert not (root / "debug" / "runs").exists()
