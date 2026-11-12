"""承認済みデータからCNN bundleとdebug可視化を生成する実行起点。

責務:
    設定を読み込み、各処理を組み立てて実行し、結果を終了コードで返す。
    画像処理・DB・モデルの詳細は担わない。

処理の流れ:
    1. 設定（configs/default.yaml と環境変数）を読み込む。
    2. 全session DBから is_dino_ready=1 の物体を読む。
    3. 有効な入力があれば、出力versionを予約する。
    4. DINOv3抽出器を作り、抽出と公開を実行して件数を報告する。

関連ファイル:
    settings.py、usecase/delivery.py、infrastructure/session_repository.py、
    infrastructure/dataset_version.py、infrastructure/dinov3_extractor.py。

設計意図:
    引数なしで実行・デバッグ（F5）できるよう、切り替えたい値は設定ファイルか環境変数で渡す。
    承認済み物体が無い場合は、重いモデルを読み込む前に終了する。

Note:
    実行には、facebookresearch/dinov3 のclone（repo_dir）と、Meta配布の重み（weights_path）が必要。
"""

from __future__ import annotations

import logging

from dinov3_pipeline.common.constants import EXIT_ERROR, EXIT_OK, EXIT_PARTIAL_FAILURE
from dinov3_pipeline.common.models import DataContractError
from dinov3_pipeline.data_loading.session_repository import SessionSam3Repository
from dinov3_pipeline.dinov3.extractor import Dinov3FeatureExtractor
from dinov3_pipeline.saving.dataset_version import reserve_dataset_version
from dinov3_pipeline.settings import load_settings
from dinov3_pipeline.usecase.delivery import run_delivery

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """承認済みデータの抽出と送付物の生成を実行する。

    Returns:
        終了コード。全件成功は0、一部失敗は1、実行できなかった場合は2。
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        settings = load_settings()
        repository = SessionSam3Repository(
            settings.data_root, settings.sqlite_timeout_seconds
        )
        loaded = repository.load_ready_samples()
        # 対象が無いのにモデルを読み込むと時間とGPUを無駄にするため、先に終了する
        if not loaded.samples:
            LOGGER.warning(
                "is_dino_ready=1の有効な入力がない。SAM3で画像承認と物体別採用を確認する"
            )
            return EXIT_ERROR

        dataset_version = reserve_dataset_version(
            settings.delivery_root, settings.dataset_version
        )
        LOGGER.info("delivery version: %s", dataset_version)
        extractor = Dinov3FeatureExtractor(
            settings.repo_dir,
            settings.model_name,
            settings.weights_path,
            settings.pretrain_dataset,
            settings.crop_size,
            settings.device,
        )
        summary = run_delivery(
            loaded,
            repository,
            extractor,
            settings.delivery_root,
            settings.crop_size,
            dataset_version,
            settings.mask_depth,
        )
    except (DataContractError, OSError, ValueError, RuntimeError):
        LOGGER.exception("DINOv3のdeliveryが失敗した")
        return EXIT_ERROR

    LOGGER.info(
        "delivery %s: 完了%s/%s、PCA%s件、issue%s件",
        dataset_version,
        summary.completed_count,
        summary.eligible_count,
        summary.output.visualization_count,
        summary.output.issue_count,
    )
    LOGGER.info("CNN: %s", summary.output.cnn_path)
    LOGGER.info("debug: %s", summary.output.debug_path)
    return EXIT_PARTIAL_FAILURE if summary.failed_sample_ids else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
