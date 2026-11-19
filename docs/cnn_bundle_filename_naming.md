# CNN bundleファイル命名仕様

## 目的

CNN向けbundleを複数回作成した場合も、画像と物体の対応が追えるようにする。`sample_id`から生成している毎回異なるハッシュ名をやめ、元の画像IDと物体番号に基づくファイル名を使う。

## 対象と入出力

保存処理は承認済みsample、crop画像、DINOv3特徴量を受け取り、CNN bundleとdebug出力を作成する。命名変更の対象は次のとおり。

- CNN向けcrop RGB、depth、mask
- bundle内の元RGB、mask、depthコピー
- debug用PCA画像
- 最終特徴量と4つのblock特徴量の`.npy`

画像の画素値、crop、特徴量の計算、画像形式、ディレクトリ構成、dataset versionの扱いは変更しない。

## 命名規則

各sampleについて、ファイル名の共通部分（以下`asset-key`）を次のようにする。

```text
{image_id}_{object_index:04d}
```

- `image_id`は元データの画像IDを使う。
- `object_index`は元データの物体番号を使い、10進数で4桁以上にゼロ埋めする。4桁を超える場合は桁を切り詰めない。
- 拡張子は現在の各形式を維持する。例：crop RGB・mask・PCAは`.png`、crop depthは`.tiff`、特徴量は`.npy`、元画像コピーは入力ファイルの拡張子。
- `image_id`はファイル名の一部として使用する。実装前に、出力先のファイル名に安全に使える値であることを確認する。

例として、`image_id`が`image_17`、`object_index`が`0`の場合、共通部分は`image_17_0000`となる。

```text
cnn/<dataset_version>/rgb/image_17_0000.png
cnn/<dataset_version>/depth/image_17_0000.tiff
cnn/<dataset_version>/masks/image_17_0000.png
cnn/<dataset_version>/original_rgb/image_17_0000.jpg
cnn/<dataset_version>/original_masks/image_17_0000.png
cnn/<dataset_version>/original_depth/image_17_0000.tiff
cnn/<dataset_version>/features/image_17_0000.npy
cnn/<dataset_version>/feature_block_1/image_17_0000.npy
cnn/<dataset_version>/feature_block_2/image_17_0000.npy
cnn/<dataset_version>/feature_block_3/image_17_0000.npy
cnn/<dataset_version>/feature_block_4/image_17_0000.npy
debug/<dataset_version>/<capture_date>/<session_id>/pca_rgb/image_17_0000.png
```

## SQLiteとの関係

SQLiteのテーブル、列、型、制約、metadata key/valueは変更しない。`sample_id`と`image_id`の保存値も変更しない。既存の画像・特徴量path列には、新しい`asset-key`を使ったbundle相対pathを保存する。相対pathの区切りは`/`とする。

## 責務と制約

- 保存ファイル名とSQLiteのpath列を組み立てる責務は`src/dinov3_pipeline/saving/cnn_bundle_exporter.py`にある。
- 元sampleの`image_id`と`object_index`は、`src/dinov3_pipeline/data_loading/session_repository.py`で読み込まれる。`object_index`は`ReadySample.row_key`の末尾要素として保持される。
- 対象directory、出力形式、保存・公開順序、既存versionの上書き禁止は[CNN向け出力契約](./contracts/cnn_delivery.md)に従う。
- 既存bundleのファイル名を遡及変更、移動、上書きしない。新たに生成するbundleにのみ適用する。

## 受け入れ条件

1. 同一sampleに属するcrop画像、元画像コピー、PCA画像、および全5種類の`.npy`が、同じ`image_id`・`object_index`由来の`asset-key`を使う。
2. `object_index`は4桁以上のゼロ埋め表記で、ファイル名に含まれる。
3. 元画像コピーは既存の拡張子を保ち、他のファイル形式・画像内容は本仕様変更によって変わらない。
4. SQLite schemaと既存の識別子列は不変であり、path列は対応する新しいファイルを指す。
5. 新規bundleの名前が同一sampleについて再実行間で安定する。

## 実装上の検証

- `image_id`は既存の安全な名前要素パターンに合う場合だけ使用する。不正な値はbundle出力開始前に拒否し、出力先外への書き込みを防ぐ。
- 同一bundle内で同じasset名になるsampleがある場合、bundle出力開始前に拒否する。大文字・小文字だけが異なる名前も、Windows等の大文字小文字を区別しないファイルシステムでの上書きを防ぐため重複として扱う。
- `object_index`は`ReadySample.row_key`末尾の0以上の整数を使用する。4桁を超える場合もそのまま全桁を出力する。
- 拒否時に別の名前要素を追加したり、既存ファイルを上書きしたりしない。
