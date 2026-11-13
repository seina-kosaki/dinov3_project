"""承認済みデータからCNN bundleとdebug可視化を生成する実行起点。

責務:
    設定を読み込み、各処理を組み立てて実行し、結果を終了コードで返す。
    画像処理・DB・モデルの詳細は担わない。

処理の流れ:
    1. 設定（configs/default.yaml と環境変数）を読み込む。
    2. 全session DBから is_dino_ready=1 の物体を読む。
    3. 有効な入力があれば、出力versionを予約する。
    4. DINOv3抽出器を作り、抽出と公開を実行して件数を報告する。
"""

from __future__ import annotations

import logging

from dinov3_pipeline.common.constants import EXIT_ERROR, EXIT_OK, EXIT_PARTIAL_FAILURE
from dinov3_pipeline.common.models import DataContractError
from dinov3_pipeline.data_loading.session_repository import SessionSam3Repository
from dinov3_pipeline.dinov3.extractor import Dinov3FeatureExtractor
from dinov3_pipeline.saving.dataset_version import reserve_dataset_version
from dinov3_pipeline.settings import load_settings
from dinov3_pipeline.usecase.sample_processing import process_approved_samples

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """承認済みデータの抽出と送付物の生成を実行する。

    Returns:
        終了コード。全件成功は0、一部失敗は1、実行できなかった場合は2。
    """
    # 実行状況をINFO以上で記録できるよう、ログのレベルと表示形式を設定する
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        # configs/default.yamlと環境変数から、入出力先やモデルなどの実行設定を読み込む
        settings = load_settings()

        # 全session DBを読むためのrepositoryを、共有データの場所とSQLite timeoutで準備する
        repository = SessionSam3Repository(settings.data_root, settings.sqlite_timeout_seconds)

        # 各session DBから、DINOv3処理の対象として承認されたsampleを読み込む
        loaded = repository.load_ready_samples()

        # 対象が無いのにモデルを読み込むと時間とGPUを無駄にするため、先に終了する
        if not loaded.samples:
            LOGGER.warning("is_dino_ready=1の有効な入力がない。SAM3で画像承認と物体別採用を確認する")
            return EXIT_ERROR

        # 出力先で使用中の版と重ならないよう、この実行用のdataset versionを確保する
        dataset_version = reserve_dataset_version(settings.delivery_root, settings.dataset_version)
        LOGGER.info("delivery version: %s", dataset_version)
        # 指定されたモデルを読み込み、設定したcropサイズと実行デバイスで抽出器を準備する
        extractor = Dinov3FeatureExtractor(
            settings.model_name,
            settings.pretrain_dataset,
            settings.device,
        )
        # 承認sampleごとに特徴量を推論・抽出し、CNN bundleとdebug可視化を出力する
        summary = process_approved_samples(
            loaded,
            repository,
            extractor,
            settings.delivery_root,
            settings.crop_size,
            dataset_version=dataset_version,
            mask_depth=settings.mask_depth,
            mask_rgb_background=settings.mask_rgb_background,
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
    # mainの結果をプロセスの終了コードとしてOSへ返す
    raise SystemExit(main())
