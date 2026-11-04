"""特徴量抽出の境界（Protocol）を定義する。

責務:
    usecase が依存する抽出処理のインターフェースだけを定義する。実装は持たない。

処理の流れ:
    usecase が FeatureExtractor 経由で抽出を呼び、実体は infrastructure のDINOv3実装が担う。

関連ファイル:
    usecase/delivery.py、infrastructure/dinov3_extractor.py。

設計意図:
    実モデル（GPUと数百MB以上の重みが必要）を使わずに usecase を試験できるよう、
    差し替えが必要な抽出器だけを Protocol にした。他の依存は単一実装のため抽象化しない。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from dinov3_pipeline.domain.models import FeatureArtifact, ReadySample


class FeatureExtractor(Protocol):
    """承認済み物体から空間特徴量を生成する境界である。

    Note:
        テストでダミー実装へ差し替えるために存在する。差し替え不要になれば削除してよい。
    """

    def extract(self, sample: ReadySample, destination: Path) -> FeatureArtifact:
        """HWC float32 特徴量を保存し、生成元情報を返す。

        Args:
            sample: 承認済み物体。
            destination: 特徴量ファイルの保存先。

        Returns:
            保存した特徴量とモデル識別情報。
        """
        ...
