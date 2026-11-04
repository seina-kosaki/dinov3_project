# dataset_version 自動採番仕様

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

**出力:** 自動生成したversionを使い、現行の構成に従って次を別々に保存する。

- CNN: `cnn/<version>/`
- PCA debug: `debug/<日付>/<session_id>/<version>/pca_rgb/`
- 実行issue: `debug/runs/<version>/issues.csv`
- CNN SQLite: `dataset_metadata` に同じversionを記録する

## 動作

1. 実行時に、既存出力と重複しない連番versionを決める。
2. 決めたversionを、その実行のCNN、debug、issue、SQLite metadataに一貫して渡す。
3. 同じversionへの既存成果物がある場合は上書きしない。
4. CNN bundleの作成・検証が完了する前に未完成bundleを公開しない。既存の一時領域を経由した公開動作を維持する。

### 採番規則（実装確定）

- `dataset_version` の省略またはYAMLの `null` で自動採番する。既存の `capture_v` + 数字に完全一致する名前の最大番号に1を加える。初回は `capture_v001`、最低3桁とし、999の次は `capture_v1000`。欠番は埋めず、日時は使わない。
- 探索範囲は、同じ `delivery_root` 内の `cnn/<version>`、`debug/runs/<version>`、`debug/<日付>/<session_id>/<version>`、および `.dataset_versions/<version>` の予約記録。CNN以外だけが残る場合も使用済みとする。該当する名前のファイルも保護対象とする。内部stagingや別rootは探索しない。
- 有効な入力取得後、モデル読込前に `infrastructure/dataset_version.py` で採番し、`.dataset_versions/<version>/` を原子的な `mkdir`（既存を許可しない）で予約する。自動採番の予約競合時は最大番号を読み直して再試行し、明示指定の競合時は失敗する。同じrootを使う本実行起点同士の並行実行を対象とする。
- 予約記録は成功・失敗・プロセス終了を問わず残す。予約後のモデル読込失敗・抽出失敗・公開失敗では番号を消費し、欠番を許容する。入力なしや設定不正など予約前の終了は番号を消費しない。予約記録はCNN bundleではなく、共有対象でもない。既存成果物や予約記録を欠番回収のために削除しない。
- YAMLおよび環境変数 `DINOV3_DATASET_VERSION` による明示指定は従来どおり受け付け、安全な名前として検証する。空文字は不正。成果物または予約記録が既にある名前は、入力ありの場合にモデル読込前に拒否する。数値形式の明示名は次の自動採番にも反映する。
- 確定したversionをusecaseへ一度だけ渡し、CNN/debug/issues/SQLite metadataと完了ログで共用する。既存の一時領域経由のCNN公開処理は変更しない。

## 制約・責務

- versionの生成に伴い、抽出内容、CNN/debugの形式、入力データの扱いは変更しない。
- `settings.py` は任意version設定の読み込み・検証、`entrypoints/deliver.py` は採番予約と実行の組み立て、`infrastructure/dataset_version.py` は既存名の探索と予約、`usecase/delivery.py` は確定versionの受け渡し、`infrastructure/delivery_exporter.py` はversion別成果物の書き出し・公開を担う。
- 既存version directoryや関連成果物を置き換えない。

## 受け入れ条件

- `dataset_version` を設定せず通常実行できる。
- 連続する実行で異なる連番versionが割り当てられる。
- 既存の成果物が上書きされない。
- 1回の実行でCNN、debug、issue、SQLite metadataに記録されるversionが一致する。
- 実行失敗時に未完成のCNN bundleが公開されない。

## 運用上の前提

- 予約ディレクトリの原子的作成を提供するファイルシステムを使う。予約を使わない旧版やexporterの直接呼び出し、外部ツールによる同時書き込みまでは排他しない。
- `.dataset_versions/` を維持する。これを失うと、成果物の残らない失敗番号の履歴は復元できない。
- 外部API・SQLiteスキーマ・データの保存/読み取り可能時点は変更しない。`docs/contracts/` は現リポジトリには存在せず、本変更による契約追加・変更はない。
