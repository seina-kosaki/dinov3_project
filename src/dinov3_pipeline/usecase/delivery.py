"""承認済み物体の抽出と、CNN/debug成果物の公開を順に実行する。

責務:
    読込済みの承認済み物体について、検証→特徴量抽出→再検証を行い、
    成功した物体だけを成果物として公開する。入出力の詳細は担わない。

処理の流れ:
    1. 物体ごとに承認状態と原本を検証する。
    2. 特徴量を抽出する。
    3. 抽出中の取消・改変を検出するため、もう一度検証する。
    4. 成功した物体を export_delivery で公開する。

関連ファイル:
    infrastructure/session_repository.py、infrastructure/delivery_exporter.py、
    domain/ports.py、entrypoints/deliver.py。

設計意図:
    1物体の失敗で全体を止めず、理由を記録して続行する。承認の取消が抽出中に起きても、
    古い承認の結果を出力しないよう、抽出の前後で検証する。
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dinov3_pipeline.common.constants import FEATURE_SUFFIX, STAGING_DIR
from dinov3_pipeline.common.models import (
    DataContractError,
    DeliveryOutput,
    FeatureArtifact,
    ReadySample,
    SampleLoadResult,
)
from dinov3_pipeline.common.ports import FeatureExtractor
from dinov3_pipeline.data_loading.session_repository import SessionSam3Repository
from dinov3_pipeline.saving.delivery_exporter import export_delivery

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeliverySummary:
    """実行の対象数・完了数・失敗した物体を報告する。"""

    output: DeliveryOutput
    eligible_count: int
    completed_count: int
    failed_sample_ids: tuple[str, ...]


def run_delivery(
    loaded: SampleLoadResult,
    repository: SessionSam3Repository,
    extractor: FeatureExtractor,
    delivery_root: Path,
    crop_size: int,
    dataset_version: str,
    mask_depth: bool = True,
) -> DeliverySummary:
    """承認済み物体から特徴量を抽出し、CNN bundleとdebugを公開する。

    Args:
        loaded: 読込済みの承認済み物体と除外理由。
        repository: 抽出前後の再検証に使うsession DBのRepository。
        extractor: 特徴量抽出器。
        delivery_root: 送付root。
        crop_size: crop辺長。
        dataset_version: 出力version。
        mask_depth: Trueならdepthのmask外を0にし、Falseならcrop内の生値を保存する。

    Returns:
        公開結果と、対象・完了・失敗の件数。

    Raises:
        DataContractError: 承認済み物体が無い、または全物体の抽出に失敗した場合。
    """
    if not loaded.samples:
        raise DataContractError("is_dino_ready=1の有効なsampleがない")

    staging_root = Path(delivery_root) / STAGING_DIR
    staging_root.mkdir(parents=True, exist_ok=True)
    records: list[tuple[ReadySample, FeatureArtifact]] = []
    issues = list(loaded.issues)
    failed: list[str] = []
    # 抽出途中の特徴量は一時領域へ置き、公開は export_delivery だけに任せる
    with tempfile.TemporaryDirectory(
        prefix=f"{dataset_version}-", dir=staging_root
    ) as temporary:
        for sample in loaded.samples:
            key = hashlib.sha256(sample.sample_id.encode("utf-8")).hexdigest()
            try:
                repository.validate_sample(sample)
                artifact = extractor.extract(
                    sample, Path(temporary) / f"{key}{FEATURE_SUFFIX}"
                )
                # 抽出中に承認が取り消された物体を、古い承認のまま出力しない
                repository.validate_sample(sample)
                records.append((sample, artifact))
            except Exception as error:
                # 1物体の失敗で全体を止めないため、種類を問わず理由を記録して続行する
                LOGGER.exception("sample %sの抽出に失敗した", sample.sample_id)
                issues.append(
                    f"{sample.session_id}/{sample.image_id}: DINO抽出: {error}"
                )
                failed.append(sample.sample_id)

        if not records:
            raise DataContractError("全てのready sampleでDINO抽出に失敗した")
        output = export_delivery(
            delivery_root, crop_size, dataset_version, records, issues, mask_depth
        )

    return DeliverySummary(
        output, len(loaded.samples), output.exported_count, tuple(failed)
    )
