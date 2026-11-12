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
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from pydantic_settings import PydanticBaseSettingsSource, YamlConfigSettingsSource

from dinov3_pipeline.common.constants import HF_MODEL_IDS
from dinov3_pipeline.settings import Settings, load_settings


def test_settings_customise_sources_preserves_keyword_names_and_excludes_unused_sources() -> None:
    """overrideのキーワード名と、採用ソースの順序を維持する。"""
    init = Mock(spec=PydanticBaseSettingsSource)
    env = Mock(spec=PydanticBaseSettingsSource)
    dotenv = Mock(spec=PydanticBaseSettingsSource)
    secrets = Mock(spec=PydanticBaseSettingsSource)
    sources = Settings.settings_customise_sources(
        settings_cls=Settings,
        init_settings=init,
        env_settings=env,
        dotenv_settings=dotenv,
        file_secret_settings=secrets,
    )
    init_source, env_source, yaml_source = sources
    assert init_source is init
    assert env_source is env
    assert isinstance(yaml_source, YamlConfigSettingsSource)
    dotenv.assert_not_called()
    secrets.assert_not_called()


def test_load_settings_default_yaml_returns_positive_crop_size() -> None:
    """既定のYAMLを引数なしで読み込める。"""
    # 準備・実行
    settings = load_settings()

    # 検証
    assert settings.crop_size > 0
    assert settings.mask_depth is False
    assert settings.mask_rgb_background is True


def test_load_settings_default_version_is_automatic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DINOV3_DATASET_VERSION", raising=False)
    assert load_settings().dataset_version is None


def test_load_settings_explicit_version_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DINOV3_DATASET_VERSION", "legacy_v1")
    assert load_settings().dataset_version == "legacy_v1"
    assert load_settings(dataset_version="custom_v2").dataset_version == "custom_v2"


@pytest.mark.parametrize("mask_depth", [False, True])
def test_load_settings_env_override_replaces_mask_depth(
    monkeypatch: pytest.MonkeyPatch,
    mask_depth: bool,
) -> None:
    """環境変数でdepthのmask適用を切り替えられる。"""
    # 準備
    monkeypatch.setenv("DINOV3_MASK_DEPTH", str(mask_depth).lower())

    # 実行
    settings = load_settings()

    # 検証
    assert settings.mask_depth is mask_depth


@pytest.mark.parametrize("mask_rgb_background", [False, True])
def test_load_settings_env_override_replaces_rgb_mask_setting(
    monkeypatch: pytest.MonkeyPatch,
    mask_rgb_background: bool,
) -> None:
    """環境変数でRGB背景除去を切り替えられる。"""
    monkeypatch.setenv(
        "DINOV3_MASK_RGB_BACKGROUND", str(mask_rgb_background).lower()
    )
    assert load_settings().mask_rgb_background is mask_rgb_background


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
    model_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    monkeypatch.setenv("DINOV3_MODEL_NAME", model_id)

    # 実行
    settings = load_settings()

    # 検証
    assert settings.model_name == model_id


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


@pytest.mark.parametrize("model_id", HF_MODEL_IDS)
def test_load_settings_accepts_only_official_model_ids(model_id: str) -> None:
    assert load_settings(model_name=model_id).model_name == model_id


@pytest.mark.parametrize("name", ["repo_dir", "weights_path"])
def test_load_settings_rejects_removed_model_paths(name: str) -> None:
    with pytest.raises(ValidationError):
        load_settings(**{name: Path("unused")})


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
        ("dataset_version", ""),
        ("sqlite_timeout_seconds", 0),
        ("model_name", "dinov3_vitl16"),
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
