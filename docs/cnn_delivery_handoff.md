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

### capture単位の配置

目的は、PCA画像と実行issues.csvをcapture単位で探せるようにすること。ユーザーの`capture_ID`は既存の`dataset_version`（例：`capture_v001`）を指し、新しい識別子は追加しない。本節は新規出力に適用する。

入力は既存の`dataset_version`、sampleの`capture_date`・`session_id`、PCA画像および実行issueとし、内容は変更しない。出力先を次に変更する（日付は実行日ではなくsampleの`capture_date`）。

```text
<delivery_root>/debug/<dataset_version>/<日付>/<session_id>/pca_rgb/<既存PNG名>
<delivery_root>/debug/<dataset_version>/issues.csv
```

旧出力先は次のとおり。既存データの移動・移行・上書きは行わず、新規出力から新階層を使う。

```text
<delivery_root>/debug/<日付>/<session_id>/<dataset_version>/pca_rgb/<既存PNG名>
<delivery_root>/debug/runs/<dataset_version>/issues.csv
```

各PNGは左にcrop RGB、右に前景patchのみをPCA 3成分へ射影した特徴量可視化を配置する。既存PNG名（sample_idのSHA-256 hex + `.png`）、画像内容、特徴量、CSVの形式・記録内容は変更しない。PCAを計算できない場合もCNN特徴量を維持して理由をissues.csvへ記録する現行動作を維持し、PCA画像がなくても実行issues.csvは新階層へ保存する。

### 公開・失敗時の維持事項

- [契約正本の保存・公開節](contracts/cnn_delivery.md#保存公開読取可能時点)の順序を維持する。一時領域で書き出し、SQLiteのcommit・close、issues.csv生成とdebug出力先の存在確認後にCNNを公開し、その後PCAのディレクトリを相対pathのソート順、issues.csvを最後に移動する。
- 公開前の書き込み・出力先確認、またはCNN移動自体の失敗では新しいCNNを公開しない。CNN公開後のdebug移動失敗では、完成済みCNNと移動済みdebugを残し、未移動分は通常の例外終了時に一時領域とともに削除する。ロールバック、一括トランザクション、完了マーカーは追加しない。
- CNNの読取可能時点はCNN directoryの移動完了時点のまま。CNNの存在だけでdebug完了や実行全体成功と判断しない。CNN bundle構成・SQLite契約・公開済み成果物の非上書き方針は変更しない。

### 実装方針・受け入れ条件

変更対象の責務は`infrastructure/delivery_exporter.py`のPCA・issues出力先と公開先の整合、および`infrastructure/dataset_version.py`の使用済みversion探索である。versionの命名・予約方式は変えず、探索の詳細は[自動採番仕様](dataset_version_autonumbering.md)に従う。

1. 新規PCA画像が`debug/<dataset_version>/<日付>/<session_id>/pca_rgb/<既存PNG名>`に出力される。複数日付・sessionも同じversion配下にまとまる。
2. 実行issues.csvが`debug/<dataset_version>/issues.csv`に保存される。PCA失敗理由の記録とCNN特徴量の維持も確認する。
3. 新階層だけ、旧PCA階層だけ、旧issues階層だけが残る各ケースで使用済みversionを検出する。CNNだけ・予約記録だけが残る場合も従来どおり保護する。自動採番は使用済み最大番号の次、明示指定は重複拒否となる。
4. 既存の新旧出力を移動・上書きせず、同名出力への公開を拒否する。
5. CNNの構成・DB・特徴量・PCA画像の名前と内容が不変であり、公開順序と失敗時の残存状態が上記に一致する。

自動検証は`tests/test_delivery_exporter.py`（パス・内容・公開失敗）、`tests/test_dataset_version.py`（新旧階層の採番・重複検出）、`tests/test_deliver_entrypoint.py`（実行起点の採番連携）で行う。実データでの手動確認は別途実施する。

## 入力条件

以下は現コードの入力処理の説明であり、外部資料と照合済みの入力契約ではない。撮影側との合意を推測して補わない。

入力元は`shared_data/<日付>/<session_id>/metadata.sqlite3`である。`sam3_dino_results.is_dino_ready=1`だけを対象にし、元RGB・maskのhash、depthの存在、三画像の寸法一致を確認する。承認状態は抽出前後に再検証する。

既定crop sizeは640であり、既定モデル`dinov3_vits16`のpatch size 16で割り切れる。mask全体が設定cropに収まらないsampleはresizeで縮めず除外し、issueへ記録する。crop size/modelを変える実行は別の`dataset_version`で保存する。

このcrop処理はdepthのピクセル値と寸法を保つが、RGB/depthの撮影時位置合わせやdepth値の物理単位を保証しない。depthを学習入力に使う前に、撮影側で位置合わせと単位を確認する。
