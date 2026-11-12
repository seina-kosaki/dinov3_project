# dataset_version 自動採番仕様

> 新規debugはcapture単位の階層へ出力し、採番・重複検出では新旧両階層を探索する。変更範囲と公開・失敗時の維持事項は[受け渡し仕様のDebug節](cnn_delivery_handoff.md#debug)を参照する。それ以外の命名・予約規則は従来どおり。

モデル設定・重みhashの変更後も、既存データ保護のため、この新旧debug階層を含むversion重複検出は維持する。

## 目的

通常実行のたびに `dataset_version` を手動指定する負担をなくし、各実行の出力を固有のversion名で保存する。既存の抽出・検証・成果物形式は変更対象としない。

## 合意済み要件

- 通常実行では `configs/default.yaml` に `dataset_version` を指定しなくてよい。
- 実行ごとにversionを自動生成し、既存出力とは別のversion directoryへ出力する。
- version名は日時ではなく、`capture_v004`、`capture_v005` のような連番とする。
- 既存出力を上書きしない。
- CNN bundle、debug成果物、SQLiteのdataset metadataで同一のversion名を使う。
- 失敗時に未完成のCNN bundleを公開しない現行動作を維持する。

## 入出力と成果物

**入力:** `configs/default.yaml` の実行設定、`delivery_root` 以下の既存成果物、および現行の承認済み入力データ。`dataset_version` は通常実行の必須入力としない。

**出力:** 自動生成したversionを使い、合意済みの構成に従って次を別々に保存する。capture_IDはこのversionを指す。

- CNN: `cnn/<version>/`
- PCA debug: `debug/<version>/<日付>/<session_id>/pca_rgb/<既存PNG名>`
- 実行issue: `debug/<version>/issues.csv`
- CNN SQLite: `dataset_metadata` に同じversionを記録する

旧出力は`debug/<日付>/<session_id>/<version>/pca_rgb/`と`debug/runs/<version>/issues.csv`。既存データは移動・移行せず、新規出力から新階層を使う。

## 動作

1. 実行時に、既存出力と重複しない連番versionを決める。
2. 決めたversionを、その実行のCNN、debug、issue、SQLite metadataに一貫して渡す。
3. 同じversionへの既存成果物がある場合は上書きしない。
4. CNN bundleの作成・検証が完了する前に未完成bundleを公開しない。既存の一時領域を経由した公開動作を維持する。

### 採番規則

- `dataset_version` の省略またはYAMLの `null` で自動採番する。既存の `capture_v` + 数字に完全一致する名前の最大番号に1を加える。初回は `capture_v001`、最低3桁とし、999の次は `capture_v1000`。欠番は埋めず、日時は使わない。
- 探索範囲は、同じ `delivery_root` 内の `cnn/<version>`、新階層の`debug/<version>`、旧階層の`debug/runs/<version>`・`debug/<日付>/<session_id>/<version>`、および `.dataset_versions/<version>` の予約記録。新階層のPCA画像またはissues.csvだけ、旧階層のPCA画像またはissues.csvだけが残る場合も使用済みとする。該当する名前のファイルも保護対象とする。旧階層の探索は移行後も維持し、内部stagingや別rootは探索しない。
- 有効な入力取得後、モデル読込前に `infrastructure/dataset_version.py` で採番し、`.dataset_versions/<version>/` を原子的な `mkdir`（既存を許可しない）で予約する。自動採番の予約競合時は最大番号を読み直して再試行し、明示指定の競合時は失敗する。同じrootを使う本実行起点同士の並行実行を対象とする。
- 予約記録は成功・失敗・プロセス終了を問わず残す。予約後のモデル読込失敗・抽出失敗・公開失敗では番号を消費し、欠番を許容する。入力なしや設定不正など予約前の終了は番号を消費しない。予約記録はCNN bundleではなく、共有対象でもない。既存成果物や予約記録を欠番回収のために削除しない。
- YAMLおよび環境変数 `DINOV3_DATASET_VERSION` による明示指定は従来どおり受け付け、安全な名前として検証する。空文字は不正。成果物または予約記録が既にある名前は、入力ありの場合にモデル読込前に拒否する。数値形式の明示名は次の自動採番にも反映する。
- 確定したversionをusecaseへ一度だけ渡し、CNN/debug/issues/SQLite metadataと完了ログで共用する。既存の一時領域経由のCNN公開処理は変更しない。

## 制約・責務

- debugの配置以外は、抽出内容、CNN/debugのファイル形式・画像内容、入力データの扱いを変更しない。version名の命名・予約方式も維持する。
- `settings.py` は任意version設定の読み込み・検証、`src/main.py` は採番予約と実行の組み立て、`saving/dataset_version.py` は既存名の探索と予約、`usecase/sample_processing.py` は確定versionの受け渡し、`saving/cnn_bundle_exporter.py` はversion別成果物の書き出し・公開を担う。
- 既存version directoryや関連成果物を置き換えない。

## 受け入れ条件

- `dataset_version` を設定せず通常実行できる。
- 連続する実行で異なる連番versionが割り当てられる。
- 既存の成果物が上書きされない。
- 1回の実行でCNN、debug、issue、SQLite metadataに記録されるversionが一致する。
- 実行失敗時に未完成のCNN bundleが公開されない。
- 新規PCA画像はversion/日付/sessionの順、issues.csvは`debug/<version>/issues.csv`となる。既存出力は移動・上書きしない。
- 新debugだけ、旧PCAだけ、旧issuesだけ、CNNだけ、予約記録だけが残る各ケースで、自動採番は使用済み最大番号の次となり、明示名は重複を拒否する。数値形式以外の安全な明示名も新旧debugから重複検出する。

## 運用上の前提

- 予約ディレクトリの原子的作成を提供するファイルシステムを使う。予約を使わない旧版やexporterの直接呼び出し、外部ツールによる同時書き込みまでは排他しない。
- `.dataset_versions/` を維持する。これを失うと、成果物の残らない失敗番号の履歴は復元できない。
- 外部API・SQLiteスキーマ・データの保存/読み取り可能時点は変更しない。[CNN出力契約](contracts/cnn_delivery.md)を維持し、debug配置変更による契約正本の変更はない。
