"""プロジェクト全体で共有する固定値を集約する。

責務:
    環境や運用で変わらない固定値（出力レイアウト、画像処理の定数、終了コード）を定義する。
    環境ごとに変わる値（パス、モデル、cropサイズ）は担わず、settings.py と configs/*.yaml が扱う。

処理の流れ:
    各モジュールが必要な定数をimportして参照する。実行時の処理は持たない。

関連ファイル:
    settings.py（パス定数の参照元）、infrastructure/ 配下の各モジュール、entrypoints/deliver.py。

設計意図:
    直書きのマジック値をここに一本化し、出力仕様やDB契約の変更箇所を1か所にする。
"""

from __future__ import annotations

from pathlib import Path

# --- パス（このファイルの位置から導出する） ---
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"
ENV_PREFIX = "DINOV3_"

# --- 入力: 撮影側セッションSQLite ---
SESSION_DB_GLOB = "*/*/metadata.sqlite3"
READY_TABLE = "sam3_dino_results"
SAMPLE_ID_PREFIX = "sam3_"
REQUIRED_READY_COLUMNS = frozenset(
    {
        "session_id",
        "image_id",
        "prediction_id",
        "revision",
        "approval_id",
        "object_index",
        "review_event_id",
        "instrument_uid",
        "source_path",
        "source_sha256",
        "depth_path",
        "mask_path",
        "mask_sha256",
        "is_dino_ready",
    }
)

# --- 出力: 送付ディレクトリのレイアウト ---
CNN_DIR = "cnn"
DEBUG_DIR = "debug"
RGB_DIR = "rgb"
DEPTH_DIR = "depth"
MASK_DIR = "masks"
FEATURE_DIR = "features"
PCA_DIR = "pca_rgb"
RUNS_DIR = "runs"
STAGING_DIR = ".staging"
ISSUES_FILE = "issues.csv"
METADATA_DB = "metadata.sqlite3"
IMAGE_SUFFIX = ".png"
DEPTH_SUFFIX = ".tiff"
FEATURE_SUFFIX = ".npy"
IMAGE_FORMAT = "PNG"
DEPTH_FORMAT = "TIFF"
SAFE_NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]*"

# --- 画像・特徴量 ---
MASK_FOREGROUND = 255
RGB_BACKGROUND = (0, 0, 0)
DEPTH_BACKGROUND = 0
IMAGE_MEAN_STD_LVD1689M = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
IMAGE_MEAN_STD_SAT493M = ((0.430, 0.411, 0.296), (0.213, 0.156, 0.143))
# 事前学習データごとの正規化統計（DINOv3公式の推奨transformに合わせる）
PRETRAIN_NORMALIZATION = {
    "lvd1689m": IMAGE_MEAN_STD_LVD1689M,
    "sat493m": IMAGE_MEAN_STD_SAT493M,
}
# DINOv3公式 hubconf.py が公開するViTバックボーン（patch tokenを持つもの）
SUPPORTED_MODEL_NAMES = (
    "dinov3_vits16",
    "dinov3_vits16plus",
    "dinov3_vitb16",
    "dinov3_vitl16",
    "dinov3_vitl16plus",
    "dinov3_vith16plus",
    "dinov3_vit7b16",
)
HUB_SOURCE = "local"
PATCH_TOKENS_KEY = "x_norm_patchtokens"
FEATURE_DTYPE = "float32"
FEATURE_LAYOUT = "HWC"
FEATURE_NDIM = 3

# --- PCA可視化 ---
PCA_COMPONENTS = 3
PCA_MASK_THRESHOLD = 0.1
NORM_EPSILON = 1e-12
RANGE_EPSILON = 1e-5
COLOR_MAX = 255

# --- 終了コード ---
EXIT_OK = 0
EXIT_PARTIAL_FAILURE = 1
EXIT_ERROR = 2
