# CNN向け出力契約（正本）

## 目的・適用範囲

DINOv3側が生成しCNN側へ渡す自己完結bundleの出力契約を定める。出力実装の対応位置は移行後の`src/dinov3_pipeline/saving/delivery_exporter.py`（`SCHEMA`、`_write_samples`、`_write_database`、`_publish`）、`data_loading/preprocessing.py`、`dinov3/extractor.py`である。外部資料との照合はしておらず、CNN側との一致を保証するものではない。外部入力契約はここでは定義しない。

## 合意済みの契約差分

合意済みの契約差分は次のとおり。

- `dataset_metadata`から`weights_sha256` key/value行を削除する。`weights_sha256`は列ではなく、従来`dataset_metadata`に格納していたkeyである。
- `samples`に`feature_block_1_path`〜`feature_block_4_path`列を追加する。既存の列と`feature_path`は維持する。
- 最終特徴量`features/<sample-key>.npy`に加えて、4つのblock特徴量ファイルをbundleに追加する。ファイル構成・取得層の対応は本書に記載する。

SQLiteのテーブル数は2つのままとし、上記以外のテーブル・列・metadata keyは追加・変更しない。

## 提供側・利用側の入出力

- 提供側の内部入力：承認状態を検証したsampleとDINOv3特徴量、crop_size、dataset_version、mask_depth。提供側の出力／CNN側の入力：`<delivery_root>/cnn/<dataset_version>/`全体。
- CNN側は同梱SQLiteから識別子とbundle相対pathを読み、対応する画像・NPYを利用する。CNN側の学習結果等は本契約の対象外。
- 元session SQLite、`shared_data`、debug、内部予約記録`.dataset_versions/`は共有しない。配置を移してもbundle内の相対参照は保たれる。

## ファイル

```text
<dataset_version>/
  metadata.sqlite3
  rgb/<sample-key>.png
  depth/<sample-key>.tiff
  masks/<sample-key>.png
  features/<sample-key>.npy
  feature_block_1/<sample-key>.npy
  feature_block_2/<sample-key>.npy
  feature_block_3/<sample-key>.npy
  feature_block_4/<sample-key>.npy
```

`sample-key`はsample_idのUTF-8表現のSHA-256 hex。全特徴量ファイルで同じsample-keyを使う。ファイル参照はSQLiteに`/`区切りのbundle相対pathとして保存する。

- RGB・depth・maskは同じ座標の`crop_size × crop_size`。resize/paddingは行わず、元ファイルは変更しない。RGBはmask外0、maskは前景255／背景0のPNG。
- depthはTIFFで、元のmode・画素値を保つ（float32入力はfloat32）。通常実行の`configs/default.yaml`既定は`mask_depth: false`で、crop内のmask外も元値を保存する。`true`時だけmask外を0にする。depthはDINOv3への入力ではない。
- 通常実行はSettingsの値を渡す。`prepare_crop`・`run_delivery`・`export_delivery`を単体で呼ぶ場合の既存の既定引数は`True`であり、YAMLの既定値とは区別する。
- 全5種類のNPYは`float32`、`(Hpatch, Wpatch, C)`のHWC layout。縦横のpatch数は`crop_size / model.patch_size`、Cはモデル依存。同じbundle内のモデル名・重みは共通。
- `features/<sample-key>.npy`には最終Transformer Blockの出力にFinal LayerNormを適用したPatch Tokenを保存する。
- `feature_block_1/`〜`feature_block_4/`には、Final LayerNorm適用前のBlock出力を保存する。block数を`L`として、各blockの位置は`L/4`、`L/2`、`3L/4`、`L`。例：12層モデルは3/6/9/12、24層モデルは6/12/18/24、32層モデルは8/16/24/32。よって`feature_block_4`は最終Transformer Block出力のFinal LayerNorm適用前である。
- いずれもspecial tokenを除いたpatch tokenを保存し、`feature_path`と各`feature_block_N_path`が同sampleの対応ファイルを指す。
- depthの物理単位・撮影時のRGB/depth位置合わせは保証しない。

## SQLite

テーブルは次の2つだけ。外部キーは定義しない。

### `dataset_metadata`

| 列 | SQLite型 | 出力での必須性 | キー／DDL制約 |
|---|---|---|---|
| `key` | TEXT | 必須・非NULL | PRIMARY KEY |
| `value` | TEXT | 必須・非NULL | NOT NULL |

次の6キーを各1行保存する。値はすべてTEXT。

| key | value |
|---|---|
| `dataset_version` | 出力directoryと同じversion |
| `crop_size` | crop辺長の10進整数文字列 |
| `mask_depth` | `"true"`または`"false"`（保存時の選択値） |
| `feature_layout` | `"HWC"` |
| `feature_dtype` | `"float32"` |
| `model_name` | DINOv3モデル名 |

これは従来の7 keyから`weights_sha256`行のみを削除した差分であり、SQLiteのテーブル・列構造変更ではない。SHA-256値を別keyや列へ置き換えることも本差分に含めない。

### `samples`

| 列 | SQLite型 | 出力での必須性 | キー／DDL制約 | 内容 |
|---|---|---|---|---|
| `sample_id` | TEXT | 必須・非NULL | PRIMARY KEY | 物体instance単位の一意ID |
| `instrument_uid` | TEXT | 必須・非NULL | NOT NULL | 器具ID |
| `session_id` | TEXT | 必須・非NULL | NOT NULL | 元session ID |
| `image_id` | TEXT | 必須・非NULL | NOT NULL | 元image ID |
| `capture_date` | TEXT | 必須・非NULL | NOT NULL | 元DBが所属する日付 |
| `rgb_path` | TEXT | 必須・非NULL | NOT NULL | crop RGBへの相対path |
| `depth_path` | TEXT | 必須・非NULL | NOT NULL | crop depthへの相対path |
| `mask_path` | TEXT | 必須・非NULL | NOT NULL | crop maskへの相対path |
| `feature_path` | TEXT | 必須・非NULL | NOT NULL | 最終Patch Feature NPYへの相対path |
| `feature_block_1_path` | TEXT | 必須・非NULL | NOT NULL | `feature_block_1` NPYへの相対path |
| `feature_block_2_path` | TEXT | 必須・非NULL | NOT NULL | `feature_block_2` NPYへの相対path |
| `feature_block_3_path` | TEXT | 必須・非NULL | NOT NULL | `feature_block_3` NPYへの相対path |
| `feature_block_4_path` | TEXT | 必須・非NULL | NOT NULL | `feature_block_4` NPYへの相対path |

出力処理は全列に値を渡す。ただしDDLの`key TEXT PRIMARY KEY`と`sample_id TEXT PRIMARY KEY`には明示的な`NOT NULL`がなく、通常のSQLite rowid tableではPRIMARY KEYだけでNULL拒否を保証しない。出力での必須性とDB制約を区別する。

## 保存・公開・読取可能時点

1. 抽出前後の承認状態・原本検証を経たrecordsについて、delivery_root配下の一時directoryに画像と5種類の特徴量を保存する。
2. 保存できたsampleの行と6件のmetadataを一時bundleのSQLiteへINSERTし、commit後に接続を閉じる。この時点では公開先に未配置でありCNN側の読取対象ではない。
3. issues.csv生成とdebug出力先の存在確認後、`_publish`が完成したCNN version directoryを`cnn/<dataset_version>/`へ`Path.replace`で移す。この移動完了時点でCNNのDB・参照ファイルを読取可能になる。一時directoryは利用しない。
4. CNNの移動**後**にdebug directoryとissues.csvを順次移動する。CNN/debug全体を一括公開するトランザクションや完了マーカーはないため、CNNの存在だけでdebug公開完了・実行全体成功とは判断しない。

公開前の書き込み・出力先確認、またはCNN移動自体で失敗した場合、新しいCNN bundleは公開されず、一時directoryは通常の例外終了時に削除される。CNN移動後にdebug移動が失敗した場合は、完成済みCNNと移動済みdebugが残り、未移動のdebugは一時directoryとともに削除される（ロールバックなし）。debug directoryは相対pathのソート順、issues.csvは最後に移動する。これらは通常の例外処理の説明であり、プロセス強制終了・電源断に対する永続性は保証しない。

公開版は上書き・更新せず、別versionへ出力する。通常の採番と予約の規則は[自動採番仕様](../dataset_version_autonumbering.md)を参照する。

## 受け入れ条件

- bundleだけでSQLiteの全相対pathを解決でき、画像・5種類のNPY・全列・6 metadataの型と値が本契約に一致する。`samples`には既存`feature_path`と4つのblock pathが含まれ、全pathがbundle内の実ファイルを指す。
- `mask_depth`の記録と実際のdepth背景処理が一致する。
- 保存commit／close後にCNNを公開し、その後debugを移動する実順序と文書が一致する。外部入力契約や外部側との一致を推測で補わない。
