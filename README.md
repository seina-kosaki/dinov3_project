# DINOv3特徴量パイプライン

このプロジェクトは、`shared_data/<日付>/<session_id>/metadata.sqlite3`から`is_dino_ready=1`の器具だけを読み、リサイズなしのRGB・depth・mask cropとDINOv3の空間特徴量をCNN向けに出力する。入力SQLiteは読み取り専用で扱う。

受け渡し先向けのフォルダー構成と最小SQL schemaについては、[CNNチーム向けDINOv3受け渡し仕様](docs/cnn_delivery_handoff.md)を参照する。

## セットアップ

Python 3.12と`uv`を使用する。DINOv3のコードと重みにはMetaのDINOv3 Licenseが適用される。Hugging Faceは使わず、Metaが配布する重みを`torch.hub`のローカルソースで読み込む。

```powershell
uv sync
```

次の2つを用意する（どちらもGit管理外）。

1. 公式リポジトリをcloneする。`repo_dir`の既定値は`third_party/dinov3`である。
   ```powershell
   git clone https://github.com/facebookresearch/dinov3 third_party/dinov3
   ```
2. [Metaの配布ページ](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)でアクセスを申請し、メールのURLから重み（`.pth`）を`wget`で`weights/`へ取得する（ブラウザでは取得しない）。`dinov3_vitl16`はファイル名末尾の`-<8桁hash>.pth`を残す。

モデルは`configs/default.yaml`で切り替える。

| 項目 | 内容 |
|---|---|
| `repo_dir` | cloneした公式リポジトリ |
| `model_name` | `dinov3_vits16` / `dinov3_vits16plus` / `dinov3_vitb16` / `dinov3_vitl16` / `dinov3_vitl16plus` / `dinov3_vith16plus` / `dinov3_vit7b16` |
| `weights_path` | 上記モデルの重みファイル |
| `pretrain_dataset` | 重みの事前学習データ。`lvd1689m`（Web画像）または`sat493m`（衛星画像）。正規化統計を選ぶ |

`model_name`と`weights_path`、`pretrain_dataset`は必ず同じモデルの組で指定する。ConvNeXt系はpatch tokenを持たないため対象外である。出力のメタデータには、モデル名と重みファイルのSHA-256を記録する。PyTorchが対応デバイスを検出するとCUDAで実行し、それ以外の場合はCPUで実行する。デバイスを明示する場合は、`device`に`cuda:0`や`cpu`などのPyTorch device stringを設定する。

推論前にPyTorchのCUDA利用可否を確認できる。

```powershell
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda)"
```

`CUDA available`が`False`でもCPU推論は実行できる。重みはローカルファイルから読むため、推論時のネットワーク接続は不要である。

## 実行手順

重みと公式リポジトリを用意し、`configs/default.yaml`を設定してから、承認済みデータの抽出と送付物生成を一度に実行する。

```powershell
uv run python src/dinov3_pipeline/entrypoints/deliver.py
```

引数は不要である（VS Codeでは、このファイルを開いてF5で実行・デバッグできる）。出力versionやモデルなどの切り替えは、`configs/default.yaml`または`DINOV3_<項目名>`の環境変数（例: `DINOV3_DATASET_VERSION`）で行う。

通常の入力は`shared_data/<日付>/<session_id>/metadata.sqlite3`である。全session DBを読み取り専用で走査し、`sam3_dino_results.is_dino_ready=1`のみを対象にする。抽出前後に承認状態と原本hashを再確認する。

使用モデルは`configs/default.yaml`の`model_name`・`weights_path`・`pretrain_dataset`で指定する。`crop_size`は固定正方形cropの辺長で、DINOv3のpatch sizeで割り切れる必要がある。既定値640は、確認済みready maskの最大581×577を収める値である。対象maskが指定cropを超えるサンプルは縮小せず、debug issueへ記録して除外する。

## 出力仕様

送付先は`delivery_root`で設定し、既定値は`C:\work\SPD\dinov3_delivery`である。同じ`dataset_version`への再出力は上書きせずエラーにするため、再実行時はversionを変える。

```text
dinov3_delivery/
  cnn/<dataset_version>/
    metadata.sqlite3
    rgb/       # mask外を0にしたlossless PNG crop
    depth/     # 元値・float32 modeを保つTIFF crop
    masks/     # 二値PNG crop
    features/  # DINOv3 float32 HWC NPY
  debug/<日付>/<session_id>/<dataset_version>/pca_rgb/
    <sample>.png  # crop RGBとPCA-RGB特徴量の比較
  debug/runs/<dataset_version>/issues.csv
```

RGB・depth・maskはmaskのforeground boundsを含む同じcrop座標で切り出し、resize/paddingは行わない。crop外のRGBとdepthは0にするが、元ファイルは変更しない。depthはDINOv3へは入力せず、CNN用cropとして保存する。特徴量NPYは`float32`の`(Hpatch, Wpatch, C)`である。

CNN用SQLiteは`dataset_metadata`と`samples`の2テーブルだけである。sample行には`sample_id`、`instrument_uid`、`session_id`、`image_id`、`capture_date`と、RGB/depth/mask/featureのbundle相対pathを保存する。モデル名、重みのSHA-256、cropサイズ、特徴量layoutはdataset単位のmetadataにまとめる。出力versionは上書きしない。

## 補足

- 入力DBは日付／sessionごとに存在し、DINOv3は読み取り専用で扱う。DINO用テーブルをsource DBへ作成しない。
- CNNへ渡すのは`dinov3_delivery/cnn/<dataset_version>/`だけとし、debug画像やsource SQLiteを含む親directoryは共有しない。
- depthをCNNで使う前に、撮影時のRGB/depth位置合わせとdepth値の単位を撮影側と確認する。
- DINOv3のモデル重みとコードにはMetaのDINOv3 licenseが適用される。派生物を外部共有する前に、許可される利用・再配布の範囲をチームで確認する。
