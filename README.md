# DINOv3特徴量パイプライン

このプロジェクトは、`shared_data/<日付>/<session_id>/metadata.sqlite3`から`is_dino_ready=1`の器具だけを読み、リサイズなしのRGB・depth・mask cropとDINOv3の空間特徴量をCNN向けに出力する。入力SQLiteは読み取り専用で扱う。

受け渡し先向けのフォルダー構成とSQL schemaの正本は[CNN出力契約](docs/contracts/cnn_delivery.md)、共有手順は[CNNチーム向けDINOv3受け渡し仕様](docs/cnn_delivery_handoff.md)を参照する。

## セットアップ

Python 3.12と`uv`を使用する。DINOv3はTransformers/Hugging Faceから読み込む。モデルID、責務別の`src/`構成、および既存の入出力契約からの許可済み差分は[移行仕様](docs/huggingface_transformers_refactor.md)を参照する。

```powershell
uv sync
```

モデルIDは`configs/default.yaml`の`model_name`で切り替える。既定は`facebook/dinov3-vits16-pretrain-lvd1689m`で、選択可能なモデルは次の6つである。初回取得・キャッシュ・認証はHugging Face Hubの標準的な仕組みに従い、認証情報をリポジトリへ保存しない。

| Hugging Face model ID | hidden size |
|---|---:|
| `facebook/dinov3-vits16-pretrain-lvd1689m` | 384 |
| `facebook/dinov3-vits16plus-pretrain-lvd1689m` | 384 |
| `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | 1024 |
| `facebook/dinov3-vith16plus-pretrain-lvd1689m` | 1280 |
| `facebook/dinov3-vit7b16-pretrain-lvd1689m` | 4096 |

6モデルはいずれもpatch size 16である。ConvNeXt系は対象外。PyTorchが対応デバイスを検出するとCUDAで実行し、それ以外の場合はCPUで実行する。デバイスを明示する場合は、`device`に`cuda:0`や`cpu`などのPyTorch device stringを設定する。

推論前にPyTorchのCUDA利用可否を確認できる。

```powershell
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda)"
```

`CUDA available`が`False`でもCPU推論は実行できる。モデルの取得・キャッシュにはHugging Face Hubの標準動作が適用される。

## 実行手順

`configs/default.yaml`を設定してから、承認済みデータの抽出と送付物生成を一度に実行する。

```powershell
uv run python src/dinov3_pipeline/entrypoints/deliver.py
```

引数は不要である（VS Codeでは、このファイルを開いてF5で実行・デバッグできる）。`dataset_version` は指定不要で、通常は `capture_v001` から連番で自動採番する。モデルなどの切り替えは、`configs/default.yaml`または`DINOV3_<項目名>`の環境変数で行う。未使用のversion名を明示する場合だけ `dataset_version` または `DINOV3_DATASET_VERSION` を指定する。

通常の入力は`shared_data/<日付>/<session_id>/metadata.sqlite3`である。全session DBを読み取り専用で走査し、`sam3_dino_results.is_dino_ready=1`のみを対象にする。抽出前後に承認状態と原本hashを再確認する。

使用モデルは`configs/default.yaml`の`model_name`で指定する。`crop_size`は固定正方形cropの辺長で、DINOv3のpatch sizeで割り切れる必要がある。既定値672は、確認済みready maskの最大581×577を収める値である。対象maskが指定cropを超えるサンプルは縮小せず、debug issueへ記録して除外する。

## 出力仕様

送付先は`delivery_root`で設定し、既定値は`C:\work\SPD\dinov3_delivery`である。自動採番はCNN・debug・予約記録の最大番号+1とし、毎回別directoryへ出力する。同じversionへの明示再出力は上書きせずエラーにする。内部の `.dataset_versions/` に予約記録を残して並行実行の重複を防ぐため、予約後の失敗は欠番となる。この記録を削除しないこと。詳細は[自動採番仕様](docs/dataset_version_autonumbering.md)を参照する。

```text
dinov3_delivery/
  cnn/<dataset_version>/
    metadata.sqlite3
    rgb/       # mask外を0にしたlossless PNG crop
    depth/     # 元値・float32 modeを保つTIFF crop
    masks/     # 二値PNG crop
    features/         # 最終Block + Final LayerNormのfloat32 HWC NPY
    feature_block_1/  # 第1/4層位置のFinal LayerNorm前 NPY
    feature_block_2/  # 第2/4層位置のFinal LayerNorm前 NPY
    feature_block_3/  # 第3/4層位置のFinal LayerNorm前 NPY
    feature_block_4/  # 最終BlockのFinal LayerNorm前 NPY
  debug/<dataset_version>/<日付>/<session_id>/pca_rgb/
    <sample>.png  # crop RGBとPCA-RGB特徴量の比較
  debug/<dataset_version>/issues.csv
```

新規出力は上記のcapture単位の配置を使う。capture_IDは既存の`dataset_version`を指す。既存出力は移動・上書きせず、採番・重複検出は新旧両階層を参照する。詳細は[受け渡し仕様のDebug節](docs/cnn_delivery_handoff.md#debug)を参照する。

RGB・depth・maskはmaskのforeground boundsを含む同じcrop座標で切り出し、resize/paddingは行わない。RGBはmask外を0にする。depthは通常実行の既定`mask_depth: false`ではcrop内のmask外も元値を保ち、`true`時だけmask外を0にする。元ファイルは変更しない。depthはDINOv3へは入力せず、CNN用cropとして保存する。全特徴量NPYは`float32`の`(Hpatch, Wpatch, C)`で、`feature`は最終BlockのFinal LayerNorm適用後、`feature_block_1`〜`4`は層位置`L/4, L/2, 3L/4, L`のFinal LayerNorm適用前の出力である。sample行のSQLiteには既存の`feature_path`と4つのblock特徴量pathを保存する。

CNN用SQLiteは`dataset_metadata`と`samples`の2テーブルだけである。sample行には`sample_id`、`instrument_uid`、`session_id`、`image_id`、`capture_date`と、RGB/depth/mask/feature/feature_block_1〜4のbundle相対pathを保存する。モデル名、cropサイズ、特徴量layout等はdataset単位のmetadataにまとめる。`dataset_metadata`の`weights_sha256` keyは出力しない。特徴量directoryおよび`samples`の追加path列を含む詳細は[CNN出力契約](docs/contracts/cnn_delivery.md#合意済みの契約差分)を参照する。出力versionは上書きしない。

## 補足

- 入力DBは日付／sessionごとに存在し、DINOv3は読み取り専用で扱う。DINO用テーブルをsource DBへ作成しない。
- CNNへ渡すのは`dinov3_delivery/cnn/<dataset_version>/`だけとし、debug画像やsource SQLiteを含む親directoryは共有しない。
- depthをCNNで使う前に、撮影時のRGB/depth位置合わせとdepth値の単位を撮影側と確認する。
- DINOv3のモデル重みとコードにはMetaのDINOv3 licenseが適用される。派生物を外部共有する前に、許可される利用・再配布の範囲をチームで確認する。
