# DINOv3 Transformers移行・可読性改善仕様

## 目的

既存の承認済みデータからCNN bundleとdebug成果物を作る外部契約を維持したまま、実行経路を読みやすく整理し、DINOv3の取得・ロードをローカルcloneと`torch.hub`からHugging Face Transformersへ移す。今回の対象はpatch特徴量を持つ6つのViTモデルであり、ConvNeXtは含めない。

本書でいう「不変」は、入力選択・承認検証・採番・保存／公開動作、出力ファイル構成・SQLiteのテーブル／列構造、および特徴量のshape・dtype・layoutを指す。唯一明示的に許可された出力内容の差分は`dataset_metadata`からの`weights_sha256` key/value行の削除であり、[出力契約正本](contracts/cnn_delivery.md)に記載する。モデル実装の変更による特徴量の実数値の完全一致は要求しない。数値差が生じ得る。

## 入出力と設定

### 入力

- 現行どおり、`data_root/<日付>/<session_id>/metadata.sqlite3`を読み取り専用で走査し、`sam3_dino_results.is_dino_ready=1`の有効なsampleを対象とする。
- 現行どおり、入力列・ファイルの検証、RGB/maskのhash確認、depthの存在確認、および抽出前後の承認状態・原本の再検証を行う。
- 設定されたHugging FaceモデルIDをTransformersからロードする。初回取得・キャッシュ・認証はHugging Face Hubの標準的な仕組みに従い、認証情報をリポジトリへ保存しない。

### モデル設定

`configs/default.yaml`の`model_name`をHugging FaceのモデルIDに統一する。既定値は`facebook/dinov3-vits16-pretrain-lvd1689m`。選択可能な値と、設定コメントに記載する期待channel次元は次のとおり。

| Hugging Face model ID | hidden size | block数 |
|---|---:|---:|
| `facebook/dinov3-vits16-pretrain-lvd1689m` | 384 | 12 |
| `facebook/dinov3-vits16plus-pretrain-lvd1689m` | 384 | 12 |
| `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 | 12 |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | 1024 | 24 |
| `facebook/dinov3-vith16plus-pretrain-lvd1689m` | 1280 | 32 |
| `facebook/dinov3-vit7b16-pretrain-lvd1689m` | 4096 | 32 |

6モデルはいずれもpatch size 16。上記hidden sizeは設定コメント・テストに反映する前に、各モデルの公式Hugging Face設定とTransformersでロードしたモデルの`config.hidden_size`で照合する。実行時にもモデルのpatch sizeと出力次元を検証し、期待する形状でない出力を保存しない。block数はモデル設定から動的に取得し、model IDごとの固定block数分岐は設けない。

`repo_dir`、ローカル`weights_path`および旧hub entry名はモデル選択に使わない。依存関係・既定設定・セットアップ文書からcloneと`torch.hub`の手順を除く。モデルID以外に必要な設定を残す場合も、合意済みの入出力・推論条件を変えないこと。

## 動作と責務

実行起点は設定と処理部品を組み立て、従来の順序を維持する。モデルを使わない入力選択や採番を先に行い、対象sampleがない場合は現行どおりモデルロード前に終了する。各sampleは従来どおり個別失敗をissueへ記録して続行し、成功sampleを既存の公開処理へ渡す。

### `src/`のディレクトリ構成

既存の配布パッケージ名`dinov3_pipeline`と、`settings.py`、`entrypoints/`、`usecase/`のimport位置は保つ。処理モジュールは下記の責務ディレクトリへ移し、移動元の`infrastructure/`と`domain/`は移行後に残さない。記載したファイル名は配置案であり、薄い委譲用モジュールやクラスを追加する要件ではない。

```text
src/dinov3_pipeline/
  __init__.py
  settings.py                   # 設定の読み込み・検証（共通utilityや処理責務とは分離）
  entrypoints/
    __init__.py
    deliver.py                  # 既存の実行起点・main()
  usecase/
    __init__.py
    delivery.py                 # 入力検証→推論→再検証→保存/公開のorchestration
  data_loading/
    __init__.py
    session_repository.py       # session DB探索、read-only読込、承認sample化・再検証
    preprocessing.py            # RGB/depth/mask読込、既存条件の共通crop
  dinov3/
    __init__.py
    extractor.py                # Transformersモデルのロードとpatch特徴量抽出
  saving/
    __init__.py
    delivery_exporter.py        # crop/NPY/PCA/SQLite/issuesの一時出力と公開
    dataset_version.py          # 使用済みversion検出と予約
    pca_visualizer.py           # debug用PCA-RGB可視化の生成
  common/
    __init__.py
    constants.py                # 出力名・固定値・終了コード等の共有定数
    file_hash.py                # 入力ファイルhash等の共通処理
    models.py                   # ReadySample等の共有データ契約・DataContractError
    ports.py                    # FeatureExtractor等の共有境界（必要なProtocolのみ）
```

配置変更の対応は次のとおり。`domain/models.py`と`domain/ports.py`はそれぞれ`common/models.py`、`common/ports.py`へ移し、入力読込側・推論側・保存側が共有する型／境界として扱う。`constants.py`、`file_hash.py`も`common/`へ移す。既存の処理実装は旧配置を互換shimとして残さず、下表の配置へ移す。

| 旧配置 | 新配置 |
|---|---|
| `infrastructure/session_repository.py` | `data_loading/session_repository.py` |
| `infrastructure/preprocessing.py` | `data_loading/preprocessing.py` |
| `infrastructure/dinov3_extractor.py` | `dinov3/extractor.py` |
| `infrastructure/delivery_exporter.py` | `saving/delivery_exporter.py` |
| `infrastructure/dataset_version.py` | `saving/dataset_version.py` |
| `infrastructure/pca_visualizer.py` | `saving/pca_visualizer.py` |
| `infrastructure/file_hash.py` | `common/file_hash.py` |
| `domain/models.py` | `common/models.py` |
| `domain/ports.py` | `common/ports.py` |
| package rootの`constants.py` | `common/constants.py` |
| package rootの`settings.py`、`entrypoints/deliver.py`、`usecase/delivery.py` | 同じ位置を維持 |

importの向きは、`entrypoints → settings/common/usecase`、`usecase → data_loading/dinov3/saving/common`、各責務ディレクトリ`→ common`とする。責務ディレクトリ同士を相互依存させず、orchestrationは既存の`usecase/delivery.py`に置く。設定は既存位置の`dinov3_pipeline.settings`を維持し、entrypointが設定と各処理を組み立てる。

現行コードと移動先の役割は次のとおり。責務の切り分けは既存機能を追いやすくするためのものであり、薄いラッパーの追加や不要な抽象化は行わない。

| 責務 | 移行後の主要モジュール | 維持・整理する内容 |
|---|---|---|
| データ読み込み | `data_loading/session_repository.py`、`data_loading/preprocessing.py` | session DB探索・read-only読込、ready sample化、再検証、画像読込と共通cropを担う。DB・画像の読込条件とcrop契約を変えない。 |
| DINOv3推論 | `dinov3/extractor.py`、`common/ports.py` | Transformersモデルのロード、評価／推論モード、現行の入力正規化・mask処理、最終層および指定中間層のpatch特徴量抽出を担う。`torch.hub`由来のrepo・重み引数を除く。 |
| 保存・公開 | `saving/delivery_exporter.py`、`saving/dataset_version.py`、`saving/pca_visualizer.py` | 現行の一時出力、NPY・画像・SQLite・debug保存、予約済みversionの公開順序と失敗時動作を維持する。追加の4出力の公開上の位置づけは未決定事項を参照する。 |
| 共通utility・データ契約 | `common/` | 共有定数、hash処理、データ契約、抽出器境界を置く。特定のI/O責務やorchestrationは持たせない。 |
| 起点・設定・orchestration | `entrypoints/deliver.py`、`settings.py`、`usecase/delivery.py` | 既存の場所を維持し、起点は組立て、設定は設定読込、usecaseは処理順序の制御を担う。 |

処理経路は「設定・入力読込 → 対象のない場合の終了／version予約 → モデル推論 → sample再検証 → 保存・公開」とし、現行の承認前後検証とversion予約時点を変更しない。特に保存は特徴量だけでなく画像、SQLite、issues/debugと公開までを含む。処理の整理に伴う無関係な挙動変更は行わない。

## 特徴量・出力契約

- RGB入力、cropの決定、リサイズ／paddingをしないこと、RGBのmask外を0にすること、および既存正規化条件は現行どおりとする。depthは従来どおりモデル入力にしない。
- 既存特徴量は選択モデルの最終出力から空間patch tokenのみを得て、CLS／register等のspecial tokenを含めず、行優先のHWCへ変換する。patch token数が`(crop_size / 16) ** 2`でない出力は拒否する。
- 追加で、block数をロード済みモデルから動的に取得し、1始まりの層位置`L/4, 2L/4, 3L/4, L`の4層を取得する。対象6モデルでは各位置が整数であり、取得順にblock1〜block4とする。内部APIが0始まりのindexを要求する場合は各位置から1を引いて変換する。4 indexは重複しない。例：12層は3, 6, 9, 12、24層は6, 12, 18, 24、32層は8, 16, 24, 32（いずれも1始まりの層番号）。
- Transformersで`get_intermediate_layers(images, n=layer_indices, reshape=True, norm=True)`が利用できるかを実装前に対象モデルのAPIで確認し、利用可能なら使用する。利用できない場合は、同じ4層の正規化済みpatch token空間マップを得る出力等価のAPI／方法を選ぶ。Transformersの特定versionは要件にしない。各層はspecial tokenを除いたpatch tokenを空間マップとして取得する。
- 既存出力と追加出力はいずれもNPY、`float32`、HWC layout `(H,W,C)`とする。既定crop size 672の場合は全6モデルで空間shape `(42,42)`、一般には`(crop_size / 16, crop_size / 16)`。各層のchannel次元は選択モデルの`hidden_size`。
- 既存のfeature出力を維持したうえで、追加4出力を`feature_block1/`、`feature_block2/`、`feature_block3/`、`feature_block4/`へ保存する。各ディレクトリ内のファイル名は既存featureファイルと同じsample key/name（NPYファイル）とし、別の層番号を付加した命名は行わない。ブロック番号は上記の取得順であり、対応する1始まり層番号は一般式`L/4, 2L/4, 3L/4, L`で定義する。これらのディレクトリをCNN bundle等のどの出力rootに置くかは、公開上の位置づけと合わせて未決定とする。
- 既存feature出力にどの層を割り当てるか、従来の最終層出力を維持するか、追加4出力をCNN向け公開bundleに含めるか別のdebug／補助成果物とするかは未決定。これらを確定契約として解釈せず、未解決事項で回答を求める。READMEおよびCNN delivery contractへの反映範囲もこの回答後に確定する。
- depthの通常実行時の`mask_depth: false`を含め、crop・depth保存の既定と選択動作は現行どおりとする。
- 既存CNN bundleのパス・ファイル名・画像形式、SQLiteの2テーブル・列、debug配置、issue記録、dataset version採番、保存・公開順序を維持する。追加4出力のbundleへの包含可否は上記未決定事項による。SQLite metadataの`model_name`には選択したHugging Face model IDを記録する。
- 合意済みの契約差分として、`dataset_metadata`の`weights_sha256` key/value行を出力しない。テーブル・列の構造は維持し、SHA-256 metadata値の算出・記録を新たな要件にしない。差分の表記は[出力契約正本](contracts/cnn_delivery.md)を参照する。

## 制約・対象外

- `third_party`へのDINOv3 clone、`repo_dir`、`torch.hub`を使わない。依存関係、config、README等のセットアップ手順を同じ方針に揃える。
- ConvNeXt、SQLite schema変更、既存入力条件・承認検証・採番・保存動作の変更、特徴量shape/layout/dtype変更は対象外。追加4出力の公開上の位置づけは未決定事項であり、本仕様だけでCNN bundleの契約を変更しない。
- モデル出力の数値を旧`torch.hub`実装と完全一致させることは受け入れ条件ではない。shape・dtype・patch位置の対応は維持する。
- 実モデルの重みや外部サービスを使わない自動テストを基本とし、モデルロード・出力形式はTransformers境界をmockしたテストで検証する。実モデルでの動作確認は自動テストと区別して記録する。

## 受け入れ条件

1. 既定で`facebook/dinov3-vits16-pretrain-lvd1689m`をTransformers経由でロードし、設定から列挙した他の5モデルへ切り替えられる。未対応IDは推論前に拒否する。
2. clone、`repo_dir`、`torch.hub`に依存せず、依存関係・config・セットアップ文書が一致する。設定コメントに6つの公式IDと期待channel次元があり、次元は公式モデル設定とロード実装の値で照合されている。
3. 入力の選択条件、承認検証、採番、既存出力の保存／公開動作およびSQLiteのテーブル・列構造を維持する。追加4出力は本仕様に記載するが、公開先・bundleへの包含は未決定事項として扱う。許可されたmetadata key差分以外の既存契約変更を行わない。
4. 6モデルすべてでblock数をモデル設定から動的に取得し、model IDによるblock数固定分岐がない。各モデルで一般式から重複のない4 indexを導き、内部index変換を検証する（12層: 3/6/9/12、24層: 6/12/18/24、32層: 8/16/24/32。表示は1始まり）。
5. 4層の特徴量が取得順にblock1〜block4として個別に得られ、各層が空間patch mapである。Transformersの指定APIの利用可否を対象モデルで確認し、利用不可時は出力等価の方法を検証する。
6. NPYが`float32` HWC `(H,W,C)`、空間shapeが`crop_size / 16`、channelが選択モデルのhidden sizeである。4出力のディレクトリ、既存featureと同じsample key/name、余分な層番号を付けない命名を確認する。6モデルすべてpatch size 16として検証する。特徴量の数値一致は要求しない。
7. 既存feature出力との関係（既存featureへ割り当てる層／従来の最終出力維持の有無）と追加4出力のCNN公開bundleへの包含可否が、ユーザー回答で確定されるまで未決定として扱われている。決定後はREADMEとCNN delivery contractへの影響を評価し、合意内容に合わせる。
8. `src/dinov3_pipeline/`が責務別に明瞭で、data loading、DINOv3推論、保存が対応ディレクトリに分かれ、共通utilityが独立している。残るentrypoint・設定・orchestrationの配置とimport関係も本仕様に一致し、薄いラッパーで複雑化していない。
9. 既存のentrypoint `dinov3_pipeline.entrypoints.deliver:main`と直接実行手順が移動後も動作し、CLIのimport先も維持されている。既存テストの内部module importは新配置へ更新され、テストを含むimportが解決する。
10. 外部入出力・DB schema・許可済み`dataset_metadata` key差分を契約文書と照合し、受け入れ可能である。関連テストと文書が更新され、少なくとも設定検証、Transformersモデルロード／特徴抽出、4層の取得順・index・個別保存、HWC shape・dtype・patch配置、既存出力との関係、entrypoint/usecase、delivery exporterの回帰を扱う。

## テスト・文書更新範囲

- テスト: `tests/test_dinov3_extractor.py`（Transformersロード、ID選択、patch size・出力次元・特殊token除外、4層の深度・index・取得順・HWC保存）、`tests/test_feature_layout.py`、`tests/test_settings.py`、`tests/test_deliver_entrypoint.py`、`tests/test_delivery_usecase.py`、`tests/test_delivery_exporter.py`。4個の追加NPYのpath、sample key/name、shape/dtype、および既存出力の維持を検証する。必要な場合はダミーTransformers出力を使い、実モデル取得を単体テストの前提にしない。
- 設定・依存: `configs/default.yaml`、`pyproject.toml`およびlockfile。
- 利用者向け文書: `README.md`、`docs/structure.md`。本仕様と矛盾するclone・ローカル重み・旧設定名・旧モデル名・既定cropの説明を見直す。
- 契約文書: `docs/contracts/cnn_delivery.md`はDBのテーブル・列構造と、許可済みの`dataset_metadata` key削除を正確に記述する。構造変更は行わない。追加4出力を公開bundleへ含めるかは未決定のため、本仕様の更新だけで正本のファイル一覧を書き換えない。回答後、必要な場合に正本・README・`docs/cnn_delivery_handoff.md`を整合させる。

## 未解決事項

- 既存`features/<sample-key>.npy`に割り当てる層を決めること。追加4層の1つを既存featureにも使うのか、既存の最終層出力を別途維持するのか、ユーザー回答を得るまで確定しない。
- `feature_block1/`〜`feature_block4/`の追加出力をCNN向け公開bundleに含めるか、別debug／補助成果物とするかを決めること。これに伴いREADMEの共有対象説明およびCNN delivery contractのbundle構成・SQLite参照方法を更新する必要があるかも回答後に判断する。現契約のSQLiteにはfeature pathが1列のみであるため、bundleに含める決定をした場合もschemaを変更する前提にはせず、参照方法を別途明確化する。
- Transformers APIの実現方法は特定versionを固定せず、対象モデルで`get_intermediate_layers`の利用可否を確認してから決定する。モデルrevision pinningは要件にしない。
