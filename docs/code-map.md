# コードマップ

- `src/main.py`: 設定と処理部品を組み立て、承認済みデータの処理を起動する。
- `src/test/run_dinov3_pca_demo.py`: 指定画像を16の倍数にリサイズしてDINOv3で推論し、5種類の特徴量をPCA可視化する。
- `src/dinov3_pipeline/settings.py`: YAML・環境変数から実行設定を読み込み、型と項目ごとの条件を検証する。
- `src/test/test_self_and_cls.py`: `self`・`cls`・Pydanticのフィールド検証をpytestと直接実行デモで学ぶ。
- `configs/default.yaml`: 入出力先、モデル、cropサイズなどの既定設定を定義する。
- `docs/learning-notes.md`: PythonやPydanticの初出構文・機能の学習メモを記録する。
- `docs/structure.md`: プロジェクト全体の構成と責務を説明する。
- `docs/dinov3_inference_api.md`: DINOv3推論APIの入出力、特徴層の選択、責務境界を定義する。
