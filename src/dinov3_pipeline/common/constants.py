"""プロジェクト全体で共有する固定値を集約する。"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"
ENV_PREFIX = "DINOV3_"

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

MASK_FOREGROUND = 255
RGB_BACKGROUND = (0, 0, 0)
DEPTH_BACKGROUND = 0
IMAGE_MEAN_STD_LVD1689M = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
IMAGE_MEAN_STD_SAT493M = ((0.430, 0.411, 0.296), (0.213, 0.156, 0.143))
PRETRAIN_NORMALIZATION = {
    "lvd1689m": IMAGE_MEAN_STD_LVD1689M,
    "sat493m": IMAGE_MEAN_STD_SAT493M,
}

HF_MODEL_IDS = (
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
)
SUPPORTED_MODEL_NAMES = HF_MODEL_IDS

HUB_SOURCE = "hf"
PATCH_TOKENS_KEY = "x_norm_patchtokens"
FEATURE_DTYPE = "float32"
FEATURE_LAYOUT = "HWC"
FEATURE_NDIM = 3

PCA_COMPONENTS = 3
PCA_MASK_THRESHOLD = 0.1
NORM_EPSILON = 1e-12
RANGE_EPSILON = 1e-5
COLOR_MAX = 255

EXIT_OK = 0
EXIT_PARTIAL_FAILURE = 1
EXIT_ERROR = 2
