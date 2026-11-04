# 構成

承認済みの器具データからDINOv3特徴量を作り、CNN向けbundleとdebug可視化を出力する。

```text
configs/default.yaml        環境・運用で変わる値（パス、dataset_version、モデル名・重み、crop_size）
weights/                    DINOv3の重み（Git管理外）
third_party/dinov3/         facebookresearch/dinov3 のclone（Git管理外、torch.hubのlocalソース）
src/dinov3_pipeline/
  constants.py              固定値（出力レイアウト、画像処理の定数、終了コード）
  settings.py               pydantic-settings で YAML と環境変数を読み込む
  entrypoints/deliver.py    実行起点。引数なしで実行・デバッグできる
  usecase/delivery.py       検証→抽出→再検証→公開の流れ
  domain/                   データ契約（models.py）と抽出器の境界（ports.py）
  infrastructure/           外部に触れる処理
    session_repository.py   session SQLite の読み取り（承認状態・原本hashの検証）
    preprocessing.py        RGB・depth・maskの共通crop（画像ファイルのみ読む）
    dinov3_extractor.py     DINOv3モデルの実行（torch.hub のローカル重み、GPU）
    file_hash.py            ファイルのSHA-256（原本・重みの同一性確認）
    pca_visualizer.py       PCA-RGB画像の生成
    delivery_exporter.py    CNN bundle・debug・SQLite の書き出しと公開
tests/                      pytest（実モデルは使わない）
```

依存の向きは `entrypoints → usecase → infrastructure → domain` の一方向である。

## 判断の記録

- `domain/ports.py` の `FeatureExtractor` だけを `Protocol` にした。GPUと重みファイルが要る抽出器は、テストでダミーへ差し替える必要があるため。他は実装が1つなので抽象化しない。
- 旧来のmanifest登録・単一DB・argparse CLI（`interface/`、`bundle_exporter.py` ほか）は、現行の入力契約と合わず機能していなかったため削除した。
- `configs/default.json` は `configs/default.yaml` へ移した（設定を pydantic-settings で読み込む方針に合わせるため）。
- モデルの読込を Hugging Face Transformers から torch.hub のローカル重みへ変えた（認証トークンと transformers 依存をなくすため）。モデル種別と重みは設定で切り替える。
