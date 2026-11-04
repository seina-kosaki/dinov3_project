# CNNチーム向けDINOv3受け渡し仕様

CNN出力契約の正本は[contracts/cnn_delivery.md](contracts/cnn_delivery.md)。本書は共有手順・debug・現行入力処理の案内として残す。外部資料がないため外部側との一致は未判定。

## 共有対象

CNNチームへ渡すのは、次のversion directoryだけである。

```text
C:\work\SPD\dinov3_delivery\cnn\<dataset_version>/
```

`shared_data`、debug directory、元のsession SQLiteは共有しない。出力は自己完結しており、ファイル参照はbundle相対pathで記録される。

通常の `dataset_version` は `capture_v001` からの連番で自動採番し、CNN・debug・SQLite metadataで共用する。既存成果物と内部予約記録の最大番号+1を使い、失敗した予約番号は欠番とする。`.dataset_versions/` は採番用の内部記録であり共有しない。明示version指定も未使用名のみ受け付ける。成果物形式・公開時点は変えない。

## 構成

ファイル構成・形式は[正本のファイル節](contracts/cnn_delivery.md#ファイル)を参照する。通常実行の`mask_depth`既定は`false`で、crop内のmask外depthも元値を保つ。`true`時だけmask外を0にする。

## SQLite

列の型・必須性・キー・metadata値は[正本のSQLite節](contracts/cnn_delivery.md#sqlite)を参照する。DB commitとcloseの後にCNN directoryを公開し、その後debugを移動する。正確な読取可能時点は[正本の保存・公開節](contracts/cnn_delivery.md#保存公開読取可能時点)を参照する。

## Debug

PCA-RGB比較画像は、CNN bundleとは別に次の階層へ作成される。

```text
C:\work\SPD\dinov3_delivery\debug\<日付>\<session_id>\<dataset_version>\pca_rgb\
```

各PNGは左にcrop RGB、右に前景patchのみをPCA 3成分へ射影した特徴量可視化を配置する。PCAを計算できない場合もCNN特徴量は維持し、理由を`debug/runs/<dataset_version>/issues.csv`に記録する。

## 入力条件

以下は現コードの入力処理の説明であり、外部資料と照合済みの入力契約ではない。撮影側との合意を推測して補わない。

入力元は`shared_data/<日付>/<session_id>/metadata.sqlite3`である。`sam3_dino_results.is_dino_ready=1`だけを対象にし、元RGB・maskのhash、depthの存在、三画像の寸法一致を確認する。承認状態は抽出前後に再検証する。

既定crop sizeは640であり、既定モデル`dinov3_vits16`のpatch size 16で割り切れる。mask全体が設定cropに収まらないsampleはresizeで縮めず除外し、issueへ記録する。crop size/modelを変える実行は別の`dataset_version`で保存する。

このcrop処理はdepthのピクセル値と寸法を保つが、RGB/depthの撮影時位置合わせやdepth値の物理単位を保証しない。depthを学習入力に使う前に、撮影側で位置合わせと単位を確認する。
