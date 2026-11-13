"""抽出器の境界を定義する。"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image


class FeatureExtractor(Protocol):
    """DINOv3 などの特徴量抽出器が満たす最小インターフェース。"""

    model_name: str

    # 学習メモ: `| None`は値が指定型かNoneであることを表す。`Optional[Image.Image]`でも書ける。
    def infer(self, rgb_image: Image.Image, mask_image: Image.Image | None = None) -> dict[str, np.ndarray]:
        """整形済みRGB画像から特徴量辞書を返す。

        Args:
            rgb_image: DINOv3へ入力する整形済みRGB画像。
            mask_image: 背景maskを入力tensorにも適用する場合のmask。

        Returns:
            特徴層名をkey、HWC配列をvalueとする辞書。
        """
        ...
