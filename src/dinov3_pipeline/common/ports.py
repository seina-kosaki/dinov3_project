"""抽出器の境界を定義する。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from dinov3_pipeline.common.models import FeatureArtifact, ReadySample


class FeatureExtractor(Protocol):
    def extract(self, sample: ReadySample, destination: Path) -> FeatureArtifact:
        ...
