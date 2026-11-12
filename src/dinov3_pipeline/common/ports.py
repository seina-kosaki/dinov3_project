"""抽出器の境界を定義する。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from dinov3_pipeline.common.models import FeatureArtifact, ReadySample


class FeatureExtractor(Protocol):
    """DINOv3 などの特徴量抽出器が満たす最小インターフェース。"""

    def extract(self, sample: ReadySample, destination: Path) -> FeatureArtifact:
        """サンプルから特徴量を抽出し、保存先に書き出す。"""
        ...
