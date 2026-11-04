---
applyTo: "tests/**"
---

- pytest を使い、実行は `uv run pytest`。テスト中は ruff / mypy を実行しない（動作確認に集中する）。
- 命名は `test_<対象>_<条件>_<期待結果>`。AAA（Arrange/Act/Assert）を空行かコメントで区切る。
- 同じロジックの複製は `@pytest.mark.parametrize` にまとめる。1テスト1検証。
- 実機・モデル・外部APIは、ダミー実装またはモックで置き換える。重い推論・GPU 依存のテストはマーカーで分離する。
- 内部実装の詳細に依存したモックは避ける。
