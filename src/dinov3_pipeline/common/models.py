"""データ契約と出力契約を定義する。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class DataContractError(ValueError):
    """境界で不正入力を拒否する例外。"""


@dataclass(frozen=True)
class ReadySample:
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
        if not self.sample_id or not self.instrument_uid:
            raise DataContractError("sample_idとinstrument_uidは必須である")


@dataclass(frozen=True)
class SampleLoadResult:
    samples: tuple[ReadySample, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True)
class FeatureArtifact:
    path: Path
    shape: tuple[int, int, int]
    dtype: str
    model_name: str
    weights_sha256: str


@dataclass(frozen=True)
class DeliveryOutput:
    cnn_path: Path
    debug_path: Path
    exported_count: int
    visualization_count: int
    issue_count: int
