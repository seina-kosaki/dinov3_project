"""configs/default.yaml と環境変数から実行設定を読み込む。

責務:
    環境・運用で変わる値（入出力パス、データセット版、モデル、cropサイズ、depthのmask有無）を型付きで提供する。
    固定値は担わず、constants.py に置く。

処理の流れ:
    環境変数（DINOV3_ 接頭辞）を優先し、無い項目を YAML から補って検証する。
    パスは設定ファイルの位置を基準に絶対パスへ解決する。

関連ファイル:
    constants.py、configs/default.yaml、entrypoints/deliver.py。

設計意図:
    方針どおり pydantic-settings で読み込み、起点を引数なしで実行できるようにする。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from dinov3_pipeline.common.constants import (
    CONFIG_DIR,
    DEFAULT_CONFIG_PATH,
    ENV_PREFIX,
    LEGACY_MODEL_ALIASES,
    PRETRAIN_NORMALIZATION,
    PROJECT_ROOT,
    SAFE_NAME_PATTERN,
    SUPPORTED_MODEL_NAMES,
)


class Settings(BaseSettings):
    """実行設定。モデルとcropサイズは設定だけで切り替えられる。

    Note:
        dataset_version が None の場合、実行起点が入力取得後に連番を予約する。
        crop_size がモデルの patch size で割り切れるかは、モデル読込後に抽出器が検証する。
        repo_dir と weights_path の存在も、抽出器がモデル読込時に確認する。
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        yaml_file=DEFAULT_CONFIG_PATH,
        yaml_file_encoding="utf-8",
        frozen=True,
    )

    data_root: Path
    delivery_root: Path
    dataset_version: str | None = Field(default=None, pattern=f"^{SAFE_NAME_PATTERN}$")
    repo_dir: Path
    model_name: str
    weights_path: Path
    pretrain_dataset: str
    device: str | None = None
    crop_size: int = Field(gt=0)
    mask_depth: bool
    sqlite_timeout_seconds: float = Field(gt=0)

    @field_validator("data_root", "delivery_root", "repo_dir", "weights_path")
    @classmethod
    def _resolve_from_config_dir(cls, value: Path) -> Path:
        """相対パスを設定ディレクトリ基準の絶対パスへ解決する。

        Args:
            value: 設定値のパス。

        Returns:
            絶対パス。絶対パスで指定された場合はそのまま返る。
        """
        # 実行ディレクトリに依存せず同じ場所を指すよう、設定ファイル基準で解決する
        return (CONFIG_DIR / value).resolve()

    @field_validator("model_name")
    @classmethod
    def _check_model_name(cls, value: str) -> str:
        """公式のViTエントリ名以外を拒否する。

        Args:
            value: 設定されたモデル名。

        Returns:
            検証済みのモデル名。

        Raises:
            ValueError: 未対応のモデル名の場合。
        """
        canonical = LEGACY_MODEL_ALIASES.get(value, value)
        if value not in SUPPORTED_MODEL_NAMES and canonical not in SUPPORTED_MODEL_NAMES:
            raise ValueError(f"model_nameは{SUPPORTED_MODEL_NAMES}のいずれかである必要がある: {value}")
        return value

    @field_validator("pretrain_dataset")
    @classmethod
    def _check_pretrain_dataset(cls, value: str) -> str:
        """正規化統計を持つ事前学習データ以外を拒否する。

        Args:
            value: 設定された事前学習データ名。

        Returns:
            検証済みの名前。

        Raises:
            ValueError: 未対応の場合。
        """
        if value not in PRETRAIN_NORMALIZATION:
            raise ValueError(f"pretrain_datasetは{tuple(PRETRAIN_NORMALIZATION)}のいずれかである必要がある: {value}")
        return value

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003 - overrideの引数名を保持し、dotenvは採用しない
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003 - overrideの引数名を保持し、secret sourceは採用しない
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """引数・環境変数・YAMLの順に優先して設定を合成する。

        Args:
            settings_cls: 設定クラス。
            init_settings: コンストラクタ引数。
            env_settings: 環境変数。
            dotenv_settings: 使用しない。
            file_secret_settings: 使用しない。

        Returns:
            優先度の高い順の設定ソース。
        """
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls))


def load_settings(**overrides: Any) -> Settings:
    """設定を読み込む。引数は主にテストで上書きに使う。

    Args:
        **overrides: 上書きする設定値。

    Returns:
        検証済みの設定。

    Raises:
        pydantic.ValidationError: 設定値が不正な場合（ValueError のサブクラス）。
    """
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings(**overrides)
