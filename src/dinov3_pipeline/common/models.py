"""データ契約と出力契約を定義する。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class DataContractError(ValueError):
    """境界で不正入力を拒否する例外。"""


@dataclass(frozen=True)
class ReadySample:
    """承認済みの1サンプルに必要な識別情報とファイル境界を表す。"""

    sample_id: str
    capture_date: str
    session_id: str
    image_id: str
    instrument_uid: str
    rgb_path: Path
    mask_path: Path
    depth_path: Path
    rgb_sha256: str
    mask_sha256: str
    database_path: Path
    row_key: tuple[str, str, str, str, int]
    fingerprint: str

    def __post_init__(self) -> None:
        """ID が欠落しているサンプルは作成時に拒否する。"""
        if not self.sample_id or not self.instrument_uid:
            raise DataContractError("sample_idとinstrument_uidは必須である")


@dataclass(frozen=True)
class SampleLoadResult:
    """読み込みに成功したサンプル群と、処理対象外として記録された問題一覧。"""

    samples: tuple[ReadySample, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True)
class FeatureSet:
    """DINOv3が返す特徴量と生成モデルをまとめる。

    Attributes:
        features: 特徴層名をkey、HWCのNumPy配列をvalueとする辞書。
        model_name: 特徴量を生成したモデルID。
    """

    features: dict[str, np.ndarray]
    model_name: str


@dataclass(frozen=True)
class DeliveryOutput:
    """CNN bundle と debug 出力の生成結果を集約して返す。"""

    cnn_path: Path
    debug_path: Path
    exported_count: int
    visualization_count: int
    issue_count: int
