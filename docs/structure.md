# 構成

承認済みの器具データからDINOv3特徴量を作り、CNN向けbundleとdebug可視化を出力する。

```text
configs/default.yaml        環境・運用で変わる値（パス、任意dataset_version、モデル名・重み、crop_size）
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
    dataset_version.py      CNN/debug/予約の最大番号+1を原子的mkdirで予約
tests/                      pytest（実モデルは使わない）
```

依存の向きは `entrypoints → usecase → infrastructure → domain` の一方向である。

## 判断の記録

- `domain/ports.py` の `FeatureExtractor` だけを `Protocol` にした。GPUと重みファイルが要る抽出器は、テストでダミーへ差し替える必要があるため。他は実装が1つなので抽象化しない。
- 旧来のmanifest登録・単一DB・argparse CLI（`interface/`、`bundle_exporter.py` ほか）は、現行の入力契約と合わず機能していなかったため削除した。
- `configs/default.json` は `configs/default.yaml` へ移した（設定を pydantic-settings で読み込む方針に合わせるため）。
- モデルの読込を Hugging Face Transformers から torch.hub のローカル重みへ変えた（認証トークンと transformers 依存をなくすため）。モデル種別と重みは設定で切り替える。
- versionは入力取得後・モデル読込前に採番する。delivery_root/.dataset_versions/ の予約記録を永続化し、並行実行の競合と失敗番号の再利用を防ぐ。

## レビュー改善の合意事項・受け入れ条件

目的は診断出力・設定説明・CNN出力契約の不整合を最小限で改善すること。入力は既存の実行設定と承認済みsample、出力は既存形式のCNN bundle/debugであり、形式・採番・公開順序・設定値は変更しない。CNN出力契約の正本は[contracts/cnn_delivery.md](contracts/cnn_delivery.md)とする。

| 変更対象 | 実コード根拠と合意した動作 | 受け入れ条件 |
|---|---|---|
| `infrastructure/preprocessing.py`、`tests/test_preprocessing.py` | `_centered_crop_box`のcrop超過時の`print`2件を廃止し、既存`DataContractError`へsample ID・crop_size・mask前景の幅と高さを含める。下位処理でloggingへ置換せず、呼出境界で一度通知する方針を維持する。抽出時は`usecase/delivery.py:run_delivery`が例外ログとissueを記録し、export時は`delivery_exporter.py:_write_samples`がissueを記録する。 | 幅のみ／高さのみの超過でも上記値で原因を追える。crop関数自体は標準出力もログも出さず、例外種別・除外動作・既存issue記録を維持し、重複ログを追加しない。 |
| `configs/default.yaml` | 各実行設定に用途コメントを付ける。特に`data_root`は入力session DBの探索root、`delivery_root`はCNN/debug/予約記録の出力root、`device: null`はCUDA利用可ならCUDA・それ以外CPUの自動選択、`sqlite_timeout_seconds`は入力SQLiteのlock待機秒数（正数）。根拠は`Settings`、`Dinov3FeatureExtractor.__init__`、`SessionSam3Repository._connection`。 | 全設定の用途をコメントから理解でき、値・キー・設定優先順位・相対パス基準を変えない。 |
| `docs/contracts/cnn_delivery.md`、`docs/cnn_delivery_handoff.md`、`README.md` | `delivery_exporter.py`の`SCHEMA`・`_write_database`・`_publish`を根拠に出力契約を正本化する。既存の受け渡し文書は残し、重複仕様は正本リンクへ整理する。 | SQLite全列の型・必須性・キー、metadataの値、保存と読取可能時点がコードに一致する。CNN公開後にdebug移動が続く実順序を明記する。既存文書・見出し・リンク先を壊さず、外部入力契約は推測しない。 |
| `settings.py`、`tests/test_settings.py` | `Settings.settings_customise_sources`の未使用`dotenv_settings`／`file_secret_settings`に対するRuff ARG003を最小対応する。overrideの引数名・型・キーワード呼出互換性を保持するため、引数の改名でなく該当箇所限定の`noqa: ARG003`（不採用理由を説明）とする。返却は`init_settings → env_settings → YAML`のまま。 | 正式な引数名によるキーワード呼出が成功し、dotenv/file secretソースを返却・使用しない。対象ARG003が解消し、全体ignore・設定検証緩和など無関係な変更がない。 |
| `docs/cnn_delivery_handoff.md`、`README.md`、出力契約正本 | 実行起点が`Settings.mask_depth`を渡し、既定YAMLと`test_load_settings_default_yaml_returns_positive_crop_size`は`false`。文書の既定`true`とdepth背景の無条件0化を修正する。 | 通常実行の既定は`false`、crop内のmask外depthも元値を保持し、`true`時だけmask外を0と説明する。RGBは常にmask外0。SQLiteには選択値を記録し、関数単体の既存既定引数`True`は変更しない。 |

### 検証結果・残る確認事項

- 2026-11-04に上記5点の実装・文書更新を確認した。crop超過時の診断出力以外は、既存の処理・設定値・成果物形式を変更していない。
- 対象のpytest（preprocessing/settings/delivery usecase/delivery exporter/実行起点/自動採番）は84件成功。Mypyは設定対象の19ファイルで成功し、対象の`git diff --check`も成功した。
- 変更Python 13ファイルに対するRuffでは、対象の未使用引数に関するARG003は解消した。一方、引数数・docstring・import順序など今回の5点の範囲外の指摘40件は残っており、静的チェック全体の合格を意味しない。
- CNN出力契約はDDL・metadata生成・公開処理と照合し、通常実行の`mask_depth: false`をREADME・受け渡し文書・契約正本に反映した。
- 外部資料は未提供のため、外部側との一致・入力契約の確定は対象外。depthの物理単位や撮影時位置合わせも保証しない。
- 秘密値を含む`src/.env`は読まない。既存のワークツリー変更を巻き戻さず、この5点に限定する。
- 自動採番の従前の手動確認は完了済み。今回の変更については実モデルを使う手動確認は未完了であり、以前の確認を代用しない。通常実行で新versionへ出力できること、選択した`mask_depth`がSQLiteとdepth画像に反映されること、crop超過時に寸法情報がログ／issueに記録され直接printされないことを確認する。
