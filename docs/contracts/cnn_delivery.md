# CNN向け出力契約（正本）

## 目的・適用範囲

DINOv3側が生成しCNN側へ渡す自己完結bundleの出力契約を定める。出力実装の対応位置は`src/dinov3_pipeline/saving/cnn_bundle_exporter.py`（`SCHEMA`、`_write_samples`、`_write_database`、`_publish`）、`data_loading/preprocessing.py`、`dinov3/extractor.py`である。外部資料との照合はしておらず、CNN側との一致を保証するものではない。外部入力契約はここでは定義しない。

## 合意済みの契約差分

合意済みの契約差分は次のとおり。

- `dataset_metadata`から`weights_sha256` key/value行を削除する。`weights_sha256`は列ではなく、従来`dataset_metadata`に格納していたkeyである。
- 重みファイルに依存するhashを算出せず、そのhashを使ったbundle内の整合性確認も行わない。これは入力ファイルの既存hash検証には影響しない。
- `samples`に`feature_block_1_path`〜`feature_block_4_path`列を追加する。既存の列と`feature_path`は維持する。
- 最終特徴量`features/<sample-key>.npy`に加えて、4つのblock特徴量ファイルをbundleに追加する。ファイル構成・取得層の対応は本書に記載する。
- crop後の画像に加え、入力元のRGB・mask・depth画像を元の画素値・画像形式を保ってbundleに保存し、`samples`に各画像の相対pathを追加する。
- `samples`の`crop_x1`、`crop_y1`、`crop_x2`、`crop_y2` INTEGER列を廃止し、元画像上のcrop範囲を格納する`crop_box_json` TEXT NOT NULL列に置き換える。

SQLiteのテーブル数は2つのままとし、上記以外のテーブル・列・metadata keyは追加・変更しない。その他のファイル構成・出力内容・path列は維持する。

## 提供側・利用側の入出力

- 提供側の内部入力：承認状態を検証したsampleとDINOv3特徴量、crop_size、dataset_version、mask_depth、mask_rgb_background。提供側の出力／CNN側の入力：`<delivery_root>/cnn/<dataset_version>/`全体。
- CNN側は同梱SQLiteから識別子とbundle相対pathを読み、対応する画像・NPYを利用する。CNN側の学習結果等は本契約の対象外。
- 元session SQLite、`shared_data`、debug、内部予約記録`.dataset_versions/`は共有しない。配置を移してもbundle内の相対参照は保たれる。

## ファイル

```text
<dataset_version>/
  metadata.sqlite3
  rgb/<sample-key>.png
  depth/<sample-key>.tiff
  masks/<sample-key>.png
  original_rgb/<sample-key>.<元拡張子>
  original_depth/<sample-key>.<元拡張子>
  original_masks/<sample-key>.<元拡張子>
  features/<sample-key>.npy
  feature_block_1/<sample-key>.npy
  feature_block_2/<sample-key>.npy
  feature_block_3/<sample-key>.npy
  feature_block_4/<sample-key>.npy
```

`sample-key`はsample_idのUTF-8表現のSHA-256 hex。全特徴量ファイルで同じsample-keyを使う。ファイル参照はSQLiteに`/`区切りのbundle相対pathとして保存する。

- RGB・depth・maskは同じ座標の`crop_size × crop_size`。resize/paddingは行わず、元ファイルは変更しない。maskは前景255／背景0のPNGで、RGBは`mask_rgb_background: true`時にmask外を0にし、`false`時はcrop内の元RGBを保つ。
- `original_rgb/`、`original_masks/`、`original_depth/`には、入力元のRGB・mask・depthファイルを元の画素値・画像形式のまま保存する。元ファイルの拡張子を維持し、crop後の画像とは別ファイルとして扱う。
- 元画像上のcrop範囲は`samples.crop_box_json`にJSON objectで保存する。形式は`{"x1":16,"y1":16,"x2":48,"y2":48}`で、全値は整数。座標原点は各元画像の左上、`x1`・`y1`はcrop開始位置（含む）、`x2`・`y2`はcrop終端位置（含まない）であり、RGB・mask・depthに共通の範囲である。
- `mask_rgb_background`は通常実行で`configs/default.yaml`から設定し、`DINOV3_MASK_RGB_BACKGROUND`でも上書きできる。どちらの値でもmaskを読み込んで検証・crop位置決定に使い、mask PNGを出力する。
- depthはTIFFで、元のmode・画素値を保つ（float32入力はfloat32）。通常実行の`configs/default.yaml`既定は`mask_depth: false`で、crop内のmask外も元値を保存する。`true`時だけmask外を0にする。depthはDINOv3への入力ではない。
- `mask_rgb_background: true`時はDINOv3入力にもmaskを適用し、PCAは前景patchを対象とする。`false`時はDINOv3入力へのmask適用を省き、PCAはcrop内の全patchを対象とし、可視化のRGB panelには背景除去前のcrop RGBを使う。
- 通常実行はSettingsの値を渡す。`prepare_crop`・`process_approved_samples`・`export_cnn_bundle`・`save_pca_visualization`を単体で呼ぶ場合のmask引数の既定値は`True`であり、YAMLの既定値とは区別する。
- 全5種類のNPYは`float32`、`(Hpatch, Wpatch, C)`のHWC layout。縦横のpatch数は`crop_size / model.patch_size`、Cはモデル依存。`model_name`は選択した公式Hugging Face model IDを記録する。
- `features/<sample-key>.npy`には最終Transformer Blockの出力にFinal LayerNormを適用したPatch Tokenを保存する。
- `feature_block_1/`〜`feature_block_4/`には、Final LayerNorm適用前のBlock出力を保存する。block数を`L`として、各blockの位置は`L/4`、`L/2`、`3L/4`、`L`。例：12層モデルは3/6/9/12、24層モデルは6/12/18/24、32層モデルは8/16/24/32。よって`feature_block_4`は最終Transformer Block出力のFinal LayerNorm適用前である。
- いずれもspecial tokenを除いたpatch tokenを保存し、`feature_path`と各`feature_block_N_path`が同sampleの対応ファイルを指す。
- depthの物理単位・撮影時のRGB/depth位置合わせは保証しない。
- `mask_rgb_background`は画像の画素内容と特徴量に影響するが、bundleのファイル構成・形式・寸法・path・SQLite schemaやmetadataには追加されない。

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

これは従来の7 keyから`weights_sha256`行のみを削除した差分であり、SQLiteのテーブル・列構造変更ではない。重みhashの算出・bundle整合性確認・SHA-256値を別keyや列へ置き換えることも本差分に含めない。

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
| `original_rgb_path` | TEXT | 必須・非NULL | NOT NULL | 元RGB画像への相対path |
| `original_mask_path` | TEXT | 必須・非NULL | NOT NULL | 元mask画像への相対path |
| `original_depth_path` | TEXT | 必須・非NULL | NOT NULL | 元depth画像への相対path |
| `crop_box_json` | TEXT | 必須・非NULL | NOT NULL | 元画像左上原点のcrop座標を表すJSON object。形式は`{"x1":16,"y1":16,"x2":48,"y2":48}`で、全値は整数。`x1`・`y1`は開始位置（含む）、`x2`・`y2`は終端位置（含まない）。RGB・mask・depth共通。 |
| `feature_path` | TEXT | 必須・非NULL | NOT NULL | 最終Patch Feature NPYへの相対path |
| `feature_block_1_path` | TEXT | 必須・非NULL | NOT NULL | `feature_block_1` NPYへの相対path |
| `feature_block_2_path` | TEXT | 必須・非NULL | NOT NULL | `feature_block_2` NPYへの相対path |
| `feature_block_3_path` | TEXT | 必須・非NULL | NOT NULL | `feature_block_3` NPYへの相対path |
| `feature_block_4_path` | TEXT | 必須・非NULL | NOT NULL | `feature_block_4` NPYへの相対path |

出力処理は全列に値を渡す。ただしDDLの`key TEXT PRIMARY KEY`と`sample_id TEXT PRIMARY KEY`には明示的な`NOT NULL`がなく、通常のSQLite rowid tableではPRIMARY KEYだけでNULL拒否を保証しない。出力での必須性とDB制約を区別する。

`crop_box_json`はJSONを格納したTEXTであり、SQLite JSON functionsを利用側に必須としない。利用側は通常のJSON parserで読み取れる。

## 保存・公開・読取可能時点

1. 抽出前後の承認状態・原本検証を経たrecordsについて、delivery_root配下の一時directoryに画像と5種類の特徴量を保存する。
2. 保存できたsampleの行と6件のmetadataを一時bundleのSQLiteへINSERTし、commit後に接続を閉じる。この時点では公開先に未配置でありCNN側の読取対象ではない。
3. issues.csv生成とdebug出力先の存在確認後、`_publish`が完成したCNN version directoryを`cnn/<dataset_version>/`へ`Path.replace`で移す。この移動完了時点でCNNのDB・参照ファイルを読取可能になる。一時directoryは利用しない。
4. CNNの移動**後**にdebug directoryとissues.csvを順次移動する。CNN/debug全体を一括公開するトランザクションや完了マーカーはないため、CNNの存在だけでdebug公開完了・実行全体成功とは判断しない。

公開前の書き込み・出力先確認、またはCNN移動自体で失敗した場合、新しいCNN bundleは公開されず、一時directoryは通常の例外終了時に削除される。CNN移動後にdebug移動が失敗した場合は、完成済みCNNと移動済みdebugが残り、未移動のdebugは一時directoryとともに削除される（ロールバックなし）。debug directoryは相対pathのソート順、issues.csvは最後に移動する。これらは通常の例外処理の説明であり、プロセス強制終了・電源断に対する永続性は保証しない。

公開版は上書き・更新せず、別versionへ出力する。通常の採番と予約の規則は[自動採番仕様](../dataset_version_autonumbering.md)を参照する。

## 受け入れ条件

- bundleだけでSQLiteの全相対pathを解決でき、crop画像・元画像・5種類のNPY・全列・6 metadataの型と値が本契約に一致する。元画像は入力元と同じ画素値・画像形式を保ち、全pathがbundle内の実ファイルを指す。
- `samples`に4つのcrop座標INTEGER列がなく、代わりに`crop_box_json` TEXT NOT NULL列がある。各値は本契約のJSON object形式で、整数座標を通常のJSON parserで読み取れる。座標は元画像の左上原点で右端・下端を含まず、保存されたcropの範囲と一致し、RGB・mask・depthに同じ座標を適用する。SQLite JSON functionsの利用を要件としない。
- `mask_depth`の記録と実際のdepth背景処理が一致する。
- 保存commit／close後にCNNを公開し、その後debugを移動する実順序と文書が一致する。外部入力契約や外部側との一致を推測で補わない。
