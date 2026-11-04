"""承認済み入力と成果物の不変なデータ契約を定義する。

責務:
    DINO承認済みの1物体（ReadySample）、特徴量（FeatureArtifact）、読込結果、出力結果の型を定義する。
    DB・ファイル・モデルへのアクセスは担わない。

処理の流れ:
    infrastructure の読込処理が ReadySample を生成し、usecase が抽出と出力へ受け渡す。

関連ファイル:
    domain/ports.py、infrastructure/session_repository.py、infrastructure/delivery_exporter.py、
    usecase/delivery.py。

設計意図:
    不正な入力をモデルに渡す前に境界で拒否するため、例外とデータ契約を1か所に置く。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class DataContractError(ValueError):
    """不正または安全でない入力を、モデルや出力に渡す前に拒否する例外である。

    Note:
        承認取消や原本改変が起きたデータをCNNへ渡さないため、検証失敗は専用例外で区別する。
    """


@dataclass(frozen=True)
class ReadySample:
    """DINO承認済みの1物体と、その原本・承認状態を固定した不変データである。

    承認後に取消や原本改変が起きても検出できるよう、行の指紋（fingerprint）と
    原本のhashを保持する。不変にすることで、検証時点と抽出時点の取り違えを防ぐ。

    Attributes:
        sample_id: 承認行の指紋に由来する、物体単位の一意ID。
        capture_date: 撮影日（元DBが置かれた日付ディレクトリ名）。
        session_id: 撮影元session ID。
        image_id: 撮影元image ID。
        instrument_uid: 器具ID。
        rgb_path: 元RGB画像の絶対パス。
        mask_path: 元maskの絶対パス。
        depth_path: 元depthの絶対パス。
        rgb_sha256: 承認時点の元RGBのSHA-256。
        mask_sha256: 承認時点の元maskのSHA-256。
        database_path: 承認状態の正本であるsession DB。
        row_key: 再照合用の行キー（session_id, image_id, prediction_id, revision, object_index）。
        fingerprint: 承認行全体のSHA-256。
    """

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
        """識別子の欠落を拒否する。

        Raises:
            DataContractError: sample_id または instrument_uid が空の場合。
        """
        # 識別子が無いとCNN側で器具と画像を対応付けられないため、生成時に拒否する
        if not self.sample_id or not self.instrument_uid:
            raise DataContractError("sample_idとinstrument_uidは必須である")


@dataclass(frozen=True)
class SampleLoadResult:
    """読み込めた承認済み物体と、除外した行の理由を保持する。"""

    samples: tuple[ReadySample, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True)
class FeatureArtifact:
    """保存済みのHWC特徴量ファイルと、生成元モデルを表す。

    重みファイルのSHA-256を持たせ、同じモデル名でも重みが違えば別物として区別する。
    """

    path: Path
    shape: tuple[int, int, int]
    dtype: str
    model_name: str
    weights_sha256: str


@dataclass(frozen=True)
class DeliveryOutput:
    """CNN bundleとdebug出力の公開先、および件数を表す。"""

    cnn_path: Path
    debug_path: Path
    exported_count: int
    visualization_count: int
    issue_count: int
