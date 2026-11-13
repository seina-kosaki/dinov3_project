# DINOv3推論API仕様

## 目的

DINOv3推論を画像整形、SQLアクセス、成果物保存から分離し、他のコードから画像を直接渡して特徴量を取得できるようにする。本書は実装後の現状ではなく、合意済みの目標仕様を定める。

## 責務とデータの流れ

| 責務 | 主な入出力 | 担当 |
|---|---|---|
| SQL読み込み | 入力SQLite → 対象sampleと参照情報 | `data_loading/session_repository.py` |
| 入力データ整形 | RGB・mask・depth → crop済み画像データ | `data_loading/preprocessing.py` |
| 推論 | 整形済みRGB（必要に応じて対応するmask）→ 特徴量辞書 | `dinov3/extractor.py` |
| 保存・記録 | 整形済みデータ・特徴量・sample情報 → ファイルと出力SQLite | `saving/cnn_bundle_exporter.py` |
| 処理の組み立て | 各責務の実行順序と失敗sampleの管理 | `usecase/sample_processing.py`、`main.py` |

推論部は物体crop、画像・mask・depthの読込、出力ファイル作成、SQLiteアクセスを行わない。DINOv3向けの正規化、Tensor変換、Transformersモデル実行、Patch Tokenから特徴量配列への変換を担う。

## 推論API

### 入力

推論器は `infer` 操作を提供する。

```text
infer(rgb_image: PIL.Image, mask_image: PIL.Image | None = None)
    -> dict[str, numpy.ndarray]
```

- `rgb_image` は整形部でcrop等を済ませたRGB画像である。単体利用では、呼び出し側が用意したRGBのPIL Imageを直接渡せる。
- `mask_image` はRGB画像と同じ寸法の整形済みmaskであり、省略できる。既存のCNN出力経路でRGB背景maskをDINO入力にも適用する場合に渡す。
- crop、resize、padding、ファイル読込は推論APIの外で行う。入力はRGB modeとし、縦横はモデルのpatch sizeで割り切れること。
- 推論部は画素値をfloat32にして255で割り、DINOv3の事前学習データに対応するmean/stdで正規化する。maskが渡された場合、既存設定に従い正規化後の入力にmaskを適用する。

### 出力

返却する辞書は、次の5キーを持つ。辞書の値はすべて `float32` のNumPy配列であり、shapeは `(grid_height, grid_width, hidden_size)` のHWC形式である。

| キー | 内容 |
|---|---|
| `feature_block_1` | 第1選択層のFinal LayerNorm適用前のPatch Token |
| `feature_block_2` | 第2選択層のFinal LayerNorm適用前のPatch Token |
| `feature_block_3` | 第3選択層のFinal LayerNorm適用前のPatch Token |
| `feature_block_4` | 第4選択層のFinal LayerNorm適用前のPatch Token |
| `feature` | 最終Transformer Block出力にFinal LayerNormを適用したPatch Token |

モデルのblock数を `L` とし、中間層の選択位置は現行ルールに従う。

```text
[L // 4, 2 * L // 4, 3 * L // 4, L]
```

層番号は1始まりである。例えば12層モデルでは3/6/9/12層、24層では6/12/18/24層、32層では8/16/24/32層を選択する。モデルの設定からblock数を取得し、固定の層数に依存する分岐は行わない。各出力からCLS・register等のspecial tokenを除き、patch tokenをHWCへ並べ替える。

推論部は特徴量をファイルへ保存しない。配列をファイル化し、既存の `features/` および `feature_block_1/`〜`feature_block_4/` に配置し、出力SQLiteへpathを記録する処理は保存部が担う。返却辞書のキーと既存bundle内の特徴量名を対応させる。

## 既存出力との互換性

- CNN bundleのファイル構成、特徴量dtype・layout、`metadata.sqlite3`のschemaと値、保存・公開順序を維持する。推論部のAPI変更によって外部向けのbundle契約を変更しない。
- RGB背景maskの有効・無効を含む既存の入力処理と特徴量出力の動作は、[CNN出力契約](contracts/cnn_delivery.md)に従う。
- SQLの読み込み・承認状態の再検証と、出力SQLの書き込みは推論APIから独立させる。

## 受け入れ条件

1. 他のコードから推論器を利用し、RGBのPIL Imageを渡して5種類の特徴量を辞書で取得できる。
2. 必要な場合、RGBと寸法が一致するmaskを渡して既存のRGB背景mask動作を維持できる。maskを省略した単体推論もできる。
3. 4つの中間層はモデルblock数に応じて現行ルールで選ばれ、Final LayerNorm適用前の出力である。`feature`は最終blockのFinal LayerNorm適用後である。
4. すべての返却配列はspecial tokenを除いたPatch Tokenから作成され、`float32`・HWCである。
5. 推論部はcrop等の画像整形、ファイル保存、SQLアクセスを行わない。DINO固有の正規化とモデル推論は推論部が行う。
6. 保存部への接続後も、既存CNN bundleと出力SQLiteの形式および処理結果が維持される。
