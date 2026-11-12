# RGB mask処理とエントリーポイントの仕様

## 目的

RGB画像およびPCA可視化に対するmaskの適用を設定で切り替えられるようにする。また、entrypoint・usecase・savingの名称をそれぞれの責務に合わせて区別する。

## 設定

`configs/default.yaml` にRGB背景除去の有効・無効を選択する真偽値を追加する。既定値は `true` とし、既存どおりRGB背景をmaskで除去する。

設定キー名は `mask_rgb_background` とする。設定は既存の `Settings` を通じて読み込み、既存の `DINOV3_` 環境変数優先規則に従う。

既存の `mask_depth` は別の設定として維持し、新設定でdepth処理を切り替えない。

## 動作

両設定で、maskを読み込み、前景領域を中心にした既存の固定サイズcropを行う。前景のないmask、寸法不一致、foregroundがcropに収まらない場合の扱いも変えない。maskの前景範囲はcrop位置の決定に引き続き使用する。

| RGB背景除去 | RGB cropと特徴抽出入力 | PCA可視化 |
|---|---|---|
| 有効（既定） | 現行どおりmask外を背景色にし、DINOv3入力にもmaskを適用する。 | 現行どおり前景に対応するpatchだけでPCAを計算する。 |
| 無効 | cropからmask外を背景色に置き換える処理と、DINOv3入力へのmask適用を省略する。crop内の元RGBを保つ。 | maskでpatchを除外せず、crop内の全patchでPCAを計算する。背景除去前のRGBを可視化のRGBパネルに使う。 |

どちらの場合もmask画像は従来どおり二値PNGとして出力する。設定を無効にしてもmask画像の生成・保存、maskの検証、crop位置の計算を省略しない。

## エントリーポイントと責務別の名称

実行起点は `src/main.py` とし、現在の `src/dinov3_pipeline/entrypoints/deliver.py` は削除する。設定読込、処理部品の組み立て、usecaseの実行、終了コードと結果の報告を担う。

usecaseはsampleの検証・抽出・再検証を調整し、savingはCNN bundleとdebug成果物を書き出して公開する。責務の違いが名前から分かるよう、次の名称を採用する。

| 責務 | 現在の名称 | 採用名称 |
|---|---|---|
| 実行起点 | `entrypoints/deliver.py` | `src/main.py` |
| 承認済みsampleの処理 | `usecase/delivery.py::run_delivery` | `usecase/sample_processing.py::process_approved_samples` |
| CNN bundleの書出し・公開 | `saving/delivery_exporter.py::export_delivery` | `saving/cnn_bundle_exporter.py::export_cnn_bundle` |

entrypointを移した後も、既存の `dinov3-pipeline` コマンドを維持し、新しい起点を呼び出すようにする。配布時は `main:main` を指定する。

## 出力と互換性

mask設定の有無によって、成果物のファイル構成、形式、寸法、相対path、SQLiteのschemaは変えない。maskファイルも常に出力する。画像やPCA可視化の画素内容、および特徴量は、mask設定に応じて変わり得る。

既定の `crop_size` は672 pxであり、RGB・depth・maskは同じ672×672 px cropとなる。resizeやpaddingは行わない。PCA可視化はRGBと5種類の特徴量を3列×2行に配置し、既定では2016×1344 pxとする。別の `crop_size` を設定した場合も、既存と同じ寸法関係を維持する。

depthの背景処理は引き続き `mask_depth` のみで制御する。新設定をSQLite metadataへ追加せず、既存のCNN bundle schemaを変更しない。

実装時にRGB画像の背景処理条件が変わるため、現行のRGB内容を記載する [CNN出力契約](contracts/cnn_delivery.md) も同じ変更で更新し、設定有効時・無効時の内容を記載する。

## 受け入れ条件

1. RGB背景除去設定の既定値が有効で、既存のRGB背景除去とPCAの前景patch選択を維持する。
2. 設定無効時はRGB背景除去とDINOv3入力へのmask適用を行わず、PCAでは全patchを対象にする。
3. いずれの設定でもmaskの読込・検証・crop位置決定を維持し、mask画像を同じ形式・寸法で出力する。
4. depthの処理は従来どおり `mask_depth` に従う。
5. 設定の切替で成果物の構成・形式・寸法・path・SQLite schemaが変わらない。
6. `src/main.py` から現在の処理を実行でき、削除したentrypointを参照する実行定義が残らない。
7. entrypoint、usecase、savingの名前を責務別に区別し、上表の採用名で実行できる。

