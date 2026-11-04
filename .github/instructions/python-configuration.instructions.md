---
description: "pyproject.toml、YAML実行時設定、依存関係グループ、.env.exampleを変更するときに使用する。uv、秘密情報の保護、Pythonの起動設定を定める。"
applyTo: ["pyproject.toml", "configs/**/*.yaml", ".env.example"]
---
# Pythonプロジェクトの設定

- 依存関係と環境の管理には `uv` を使い、用途に合った依存関係グループへ `uv add` でパッケージを追加します。
- 外部ライブラリは、可能な限り提供元を確認できる公開パッケージやホイールを使い、GitHubリポジトリをcloneせずに利用します。適切な配布物がない場合は、代替方法を示してユーザーに確認します。
- このPCのCUDA環境に必要な `pyproject.toml` の `[[tool.uv.index]]` と `[tool.uv.sources]` は変更・削除しません。PyTorchを追加するときも既存の指定を維持してから `uv sync` を実行します。
- Ruff、Mypy、pytestの開発依存関係と設定は `pyproject.toml` に集約します。別の `ruff.toml` や `mypy.toml` は作成せず、設定変更時も同ファイルを更新します。
- 開発依存関係は `dev` グループに追加します。新しい環境では `uv sync` で開発依存関係も含めて同期します。
- 実行時設定は `configs/` 以下のYAMLに置き、各項目の用途をコメントで説明します。小規模な設定はまとめ、責務または変更頻度に応じて分割します。
- 秘密値は `.env` に置き、秘密値を含む `.env` ファイルをGit管理対象から除外します。`.env.example` には変数名とプレースホルダーだけを記載します。
- `argparse`、`click`、`typer` を追加したり、実行ごとのモードやログ名を要求したりしないでください。プロジェクト固有の `launch.json` なしで標準的なF5起動ができるようにします。
- AIモデルやライブラリを選定・導入する場合は公式GitHubを確認します。目的に合うHugging Face上の成果物があれば優先し、提供元との関係を確認します。
