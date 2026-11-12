# 構成

承認済みの器具データからDINOv3特徴量を作り、CNN向けbundleとdebug可視化を出力する。

```text
configs/default.yaml        実行設定（入出力root、model ID、crop等）
src/dinov3_pipeline/
  settings.py               YAML・環境変数からの設定読込
  entrypoints/deliver.py    実行起点（既存import位置を維持）
  usecase/delivery.py       入力検証・抽出・再検証・公開のorchestration
  data_loading/             session DBの読込・承認検証・共通crop
  dinov3/                   Transformersモデルのロード・特徴量推論
  saving/                   version予約、bundle/debug書出しと公開
  common/                   複数責務から使うデータ契約・境界・utility
tests/                      pytest
```

これはTransformers移行後の構成概要であり、現在の`src/`の実配置ではない。詳細なファイルツリー、旧配置からの移動先、責務とimport関係、entrypoint互換性を含む受け入れ条件は[Transformers移行・可読性改善仕様](huggingface_transformers_refactor.md#srcのディレクトリ構成)を参照する。移行後もパッケージ名`dinov3_pipeline`と`settings.py`、`entrypoints/deliver.py`、`usecase/delivery.py`の場所は維持する。DINOv3用のcloneや`torch.hub`は使わずTransformers/Hugging Faceを使う。

CNN向けファイル構成・SQLite契約の正本は[CNN出力契約](contracts/cnn_delivery.md)、version採番は[自動採番仕様](dataset_version_autonumbering.md)を参照する。

## 維持する設計判断・既存合意

- `common/ports.py`には、必要な抽出器境界`FeatureExtractor`を置く。GPU／モデル依存の推論実装をテストで差し替えるための境界であり、他の処理に薄い抽象化を追加しない。
- 旧来のmanifest登録・単一DB・argparse CLI（`interface/`、`bundle_exporter.py`等）は現行入力契約に合わないため対象外。
- 設定は`configs/default.yaml`と`pydantic-settings`を使う。
- Transformers/Hugging Faceからモデルをロードする。既定IDは`facebook/dinov3-vits16-pretrain-lvd1689m`、選択可能な6 ViTモデルと特徴量条件は[移行仕様](huggingface_transformers_refactor.md#モデル設定)を参照する。
- versionは入力取得後・モデル読込前に採番する。`delivery_root/.dataset_versions/`の予約記録で並行実行の競合と失敗番号の再利用を防ぐ。

## 既存改善合意の確認事項

既存の診断出力・設定説明改善の詳細は、過去の実装結果・検証記録として下記にまとめる。以下の契約正本・移行仕様と重複するファイル構成やmetadata key一覧はここで再掲しない。

- crop超過時は`DataContractError`にsample ID、crop_size、mask前景寸法を含める。crop関数はprintや重複ログを出さず、呼出境界でissue／ログを記録する。
- 設定の用途、優先順位、相対path基準、通常実行の`mask_depth: false`を維持する。depthは通常時にmask外の元値を保ち、`true`時のみmask外を0にする。
- CNN出力のテーブル構造および保存・公開順序は維持する。Transformers移行で許可された`dataset_metadata.weights_sha256` key/value行の削除と`samples`への4つのblock特徴量path列追加は、[出力契約正本](contracts/cnn_delivery.md#合意済みの契約差分)に明記する。
- 外部側との一致や、depthの物理単位・撮影時位置合わせは保証しない。自動採番とdebug公開の順序・失敗時動作はそれぞれの仕様書を正とする。

過去の検証結果（pytest、Mypy、Ruff等）は、その時点の変更範囲に対する記録であり、今回のディレクトリ移動やTransformers移行の検証結果を示すものではない。
