# CNNチーム向けDINOv3受け渡し仕様

## 共有対象

CNNチームへ渡すのは、次のversion directoryだけである。

```text
C:\work\SPD\dinov3_delivery\cnn\<dataset_version>/
```

`shared_data`、debug directory、元のsession SQLiteは共有しない。出力は自己完結しており、ファイル参照はbundle相対pathで記録される。

## 構成

```text
<dataset_version>/
  metadata.sqlite3
  rgb/<sample-key>.png
  depth/<sample-key>.tiff
  masks/<sample-key>.png
  features/<sample-key>.npy
```

各ファイルは同じsampleのcropである。mask外のRGBとdepthは0であり、元データは変更しない。crop時にresizeやpaddingは行わない。depth TIFFは元depthのfloat32値を保つ。

## SQLite

DBのtableは`dataset_metadata`と`samples`だけである。

`dataset_metadata(key, value)`は、dataset version、crop size、DINOv3のモデル名と重みファイルのSHA-256、feature layout、dtypeを保持する。

`samples`はCNNが必要とする次の列だけを持つ。

| 列 | 内容 |
|---|---|
| `sample_id` | 物体instance単位の一意ID |
| `instrument_uid` | 器具ID |
| `session_id` | 撮影元session ID |
| `image_id` | 撮影元image ID |
| `capture_date` | 元DBが所属する日付 |
| `rgb_path` | crop RGBへの相対path |
| `depth_path` | crop depthへの相対path |
| `mask_path` | crop maskへの相対path |
| `feature_path` | DINOv3 NPYへの相対path |

特徴量NPYは`float32`、`(Hpatch, Wpatch, C)`のHWC layoutである。モデル情報とcrop sizeは全sample共通のためdataset metadataへ集約する。

## Debug

PCA-RGB比較画像は、CNN bundleとは別に次の階層へ作成される。

```text
C:\work\SPD\dinov3_delivery\debug\<日付>\<session_id>\<dataset_version>\pca_rgb\
```

各PNGは左にcrop RGB、右に前景patchのみをPCA 3成分へ射影した特徴量可視化を配置する。PCAを計算できない場合もCNN特徴量は維持し、理由を`debug/runs/<dataset_version>/issues.csv`に記録する。

## 入力条件

入力元は`shared_data/<日付>/<session_id>/metadata.sqlite3`である。`sam3_dino_results.is_dino_ready=1`だけを対象にし、元RGB・maskのhash、depthの存在、三画像の寸法一致を確認する。承認状態は抽出前後に再検証する。

既定crop sizeは640であり、DINOv3 ViT-B/16のpatch size 16で割り切れる。mask全体が設定cropに収まらないsampleはresizeで縮めず除外し、issueへ記録する。crop size/modelを変える実行は別の`dataset_version`で保存する。

このcrop処理はdepthのピクセル値と寸法を保つが、RGB/depthの撮影時位置合わせやdepth値の物理単位を保証しない。depthを学習入力に使う前に、撮影側で位置合わせと単位を確認する。
