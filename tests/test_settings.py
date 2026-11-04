"""設定の読込（YAML・環境変数・検証）を確認する。

責務:
    Settings が既定YAML、環境変数による上書き、不正値の拒否を行うことを確認する。

処理の流れ:
    環境変数や引数で値を与え、load_settings の結果を検証する。

関連ファイル:
    src/dinov3_pipeline/settings.py、configs/default.yaml。

設計意図:
    モデルとcropサイズが設定だけで切り替えられる、という要件を回帰から守る。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dinov3_pipeline.constants import CONFIG_DIR
from dinov3_pipeline.settings import load_settings


def test_load_settings_default_yaml_returns_positive_crop_size() -> None:
    """既定のYAMLを引数なしで読み込める。"""
    # 準備・実行
    settings = load_settings()

    # 検証
    assert settings.crop_size > 0


def test_load_settings_env_override_replaces_crop_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """環境変数でcropサイズを上書きできる。"""
    # 準備
    monkeypatch.setenv("DINOV3_CROP_SIZE", "528")

    # 実行
    settings = load_settings()

    # 検証
    assert settings.crop_size == 528


def test_load_settings_env_override_replaces_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """環境変数でモデル名を切り替えられる。"""
    # 準備
    monkeypatch.setenv("DINOV3_MODEL_NAME", "dinov3_vitl16")

    # 実行
    settings = load_settings()

    # 検証
    assert settings.model_name == "dinov3_vitl16"


def test_load_settings_env_override_replaces_pretrain_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """環境変数で事前学習データ（正規化統計）を切り替えられる。"""
    # 準備
    monkeypatch.setenv("DINOV3_PRETRAIN_DATASET", "sat493m")

    # 実行
    settings = load_settings()

    # 検証
    assert settings.pretrain_dataset == "sat493m"


def test_load_settings_relative_weights_path_resolves_from_config_dir() -> None:
    """相対の重みパスを設定ファイル基準の絶対パスにする。"""
    # 準備・実行
    settings = load_settings(weights_path="../weights/custom.pth")

    # 検証
    assert settings.weights_path == (CONFIG_DIR / "../weights/custom.pth").resolve()


def test_load_settings_absolute_path_keeps_given_location(tmp_path: Path) -> None:
    """絶対パスで指定した出力先は、そのまま使う。"""
    # 準備・実行
    settings = load_settings(delivery_root=tmp_path / "delivery")

    # 検証
    assert settings.delivery_root == (tmp_path / "delivery").resolve()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("crop_size", 0),
        ("dataset_version", "../escape"),
        ("sqlite_timeout_seconds", 0),
        ("model_name", "dinov3_convnext_base"),
        ("pretrain_dataset", "unknown"),
    ],
)
def test_load_settings_invalid_value_raises_validation_error(
    name: str, value: object
) -> None:
    """不正な値は設定読込の時点で拒否する。"""
    # 準備・実行・検証
    with pytest.raises(ValidationError):
        load_settings(**{name: value})
