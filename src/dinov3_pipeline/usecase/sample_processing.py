"""承認済み物体の抽出と、CNN/debug成果物の公開を順に実行する。

責務:
    読込済みの承認済み物体について、検証→特徴量抽出→再検証を行い、
    成功した物体だけを成果物として公開する。入出力の詳細は担わない。

処理の流れ:
    1. 物体ごとに承認状態と原本を検証する。
    2. 特徴量を抽出する。
    3. 抽出中の取消・改変を検出するため、もう一度検証する。
    4. 成功した物体を export_cnn_bundle で公開する。

関連ファイル:
    data_loading/session_repository.py、saving/cnn_bundle_exporter.py、
    common/ports.py、main.py。

設計意図:
    1物体の失敗で全体を止めず、理由を記録して続行する。承認の取消が抽出中に起きても、
    古い承認の結果を出力しないよう、抽出の前後で検証する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from dinov3_pipeline.common.models import (
    DataContractError,
    DeliveryOutput,
    FeatureSet,
    ReadySample,
    SampleLoadResult,
)
from dinov3_pipeline.common.ports import FeatureExtractor
from dinov3_pipeline.data_loading.preprocessing import CroppedInputs, prepare_crop
from dinov3_pipeline.data_loading.session_repository import SessionSam3Repository
from dinov3_pipeline.saving.cnn_bundle_exporter import export_cnn_bundle

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeliverySummary:
    """実行の対象数・完了数・失敗した物体を報告する。"""

    output: DeliveryOutput
    eligible_count: int
    completed_count: int
    failed_sample_ids: tuple[str, ...]


def process_approved_samples(
    loaded: SampleLoadResult,
    repository: SessionSam3Repository,
    extractor: FeatureExtractor,
    delivery_root: Path,
    crop_size: int,
    # 学習メモ: `*`より後ろは名前付きで渡す。位置引数でも書けるが、設定値の取り違えを防ぐ。
    *,
    dataset_version: str,
    mask_depth: bool = True,
    mask_rgb_background: bool = True,
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
        mask_rgb_background: TrueならRGBのmask外を0にし、DINO入力とPCAも前景に限定する。

    Returns:
        公開結果と、対象・完了・失敗の件数。

    Raises:
        DataContractError: 承認済み物体が無い、または全物体の抽出に失敗した場合。
    """
    if not loaded.samples:
        raise DataContractError("is_dino_ready=1の有効なsampleがない")

    records: list[tuple[ReadySample, CroppedInputs, FeatureSet]] = []
    issues = list(loaded.issues)
    failed: list[str] = []
    for sample in loaded.samples:
        try:
            repository.validate_sample(sample)
            cropped = prepare_crop(sample, crop_size, mask_depth, mask_rgb_background)
            inference_mask = cropped.mask if mask_rgb_background else None
            features = extractor.infer(cropped.rgb, inference_mask)
            feature_set = FeatureSet(features, extractor.model_name)
            # 抽出中に承認が取り消された物体を、古い承認のまま出力しない
            repository.validate_sample(sample)
            records.append((sample, cropped, feature_set))
        except Exception as error:
            # 1物体の失敗で全体を止めないため、種類を問わず理由を記録して続行する
            LOGGER.exception("sample %sの抽出に失敗した", sample.sample_id)
            issues.append(f"{sample.session_id}/{sample.image_id}: DINO抽出: {error}")
            failed.append(sample.sample_id)

    if not records:
        raise DataContractError("全てのready sampleでDINO抽出に失敗した")
    output = export_cnn_bundle(
        delivery_root,
        crop_size,
        dataset_version,
        records,
        issues,
        mask_depth=mask_depth,
        mask_rgb_background=mask_rgb_background,
    )

    return DeliverySummary(
        output, len(loaded.samples), output.exported_count, tuple(failed)
    )
