import json
import re
import sys
from pathlib import PurePosixPath
from typing import Any


_SECRET_NAME = re.compile(r"^\.env(?:\..+)?$", re.IGNORECASE)
_SECRET_EXAMPLE = re.compile(r"^\.env\.example$", re.IGNORECASE)
_FILE_PATH_KEYS = {
    "file",
    "files",
    "filepath",
    "filepaths",
    "file_path",
    "file_paths",
    "path",
    "paths",
    "pathlist",
    "path_list",
    "target",
    "targets",
    "targetpath",
    "targetpaths",
    "target_path",
    "target_paths",
    "relative_path",
    "relative_paths",
    "filename",
    "filenames",
    "file_name",
    "file_names",
    "uri",
    "uris",
}
_SHELL_TOOLS = {
    "bash",
    "powershell",
    "pwsh",
    "sh",
    "zsh",
    "cmd",
    "terminal",
    "shell",
    "execute",
    "command",
    "execute_command",
    "execute_shell_command",
    "execute_shell",
    "execute_in_terminal",
    "exec",
    "exec_command",
    "run_command",
    "run_shell",
    "run_bash",
    "run_powershell",
    "run_shell_command",
    "run_in_terminal",
    "runinterminal",
    "run_terminal_cmd",
    "run_terminal_command",
    "shell_command",
    "terminal_command",
}
_FILE_TOOLS = {
    "view",
    "read",
    "read_file",
    "read_text_file",
    "read_multiple_files",
    "edit",
    "edit_file",
    "write",
    "write_file",
    "create",
    "create_file",
    "delete",
    "delete_file",
    "remove",
    "remove_file",
    "unlink",
    "file",
    "patch",
    "apply_patch",
    "applypatch",
    "replace_string_in_file",
    "multi_replace_string_in_file",
    "insert_edit_into_file",
}
_SEARCH_TOOLS = {
    "search",
    "search_files",
    "file_search",
    "search_code",
    "code_search",
    "find_files",
    "grep",
    "glob",
    "rg",
}
_PATCH_TEXT_KEYS = {"patch", "content", "changes", "replacement"}
_DELETE_PATCH = re.compile(
    r"(?im)^\s*(?:\*\*\*\s*delete file:|---\s+a/.*\n\+\+\+\s+/dev/null)"
)
_PATCH_PATH = re.compile(
    r"(?im)^\s*\*\*\*\s*(?:(?:delete|update|add)\s+file|move\s+to):\s*(.+?)\s*$"
    r"|^\s*(?:---|\+\+\+)\s+[ab]/(.+?)\s*$"
)
_PATCH_HEADER = re.compile(
    r"(?im)^[ \t]*\*\*\*[ \t]*(delete|update|add)[ \t]+file:[ \t]*(\S[^\r\n]*)$"
)
_SECRET_REFERENCE = re.compile(r"(?<![\w.-])\.env(?:\.[\w.-]+)?(?![\w.-])", re.IGNORECASE)
_SECRET_READ = re.compile(
    r"\b(?:cat|type|more|less|head|tail|gc|get-content|select-string|findstr)\b"
    r"|readalltext|readallbytes|\bopen\s*\(",
    re.IGNORECASE,
)
_SECRET_WRITE = re.compile(
    r"\b(?:del|erase|rm|remove-item|set-content|add-content|out-file|copy-item|move-item)\b"
    r"|(?:^|\s)(?:>>?|>\s*&)",
    re.IGNORECASE,
)
_SECRET_EXFIL = re.compile(
    r"\b(?:curl|wget|invoke-webrequest|invoke-restmethod|iwr|irm|scp|sftp)\b",
    re.IGNORECASE,
)
_ENV_DUMP = re.compile(
    r"(?im)(?:^\s*(?:printenv|env)\s*(?:[|;&]|$)|"
    r"\b(?:get-childitem|gci|dir)\s+env:|^\s*set\s*(?:[|;&]|$))",
    re.IGNORECASE,
)
_FILE_DELETE = re.compile(
    r"\b(?:del|erase|rm|remove-item|unlink|rmdir|rd)\b|"
    r"\b(?:git\s+)?clean\b",
    re.IGNORECASE,
)


def _decision(permission: str, reason: str) -> dict[str, Any]:
    return {
        "permissionDecision": permission,
        "permissionDecisionReason": reason,
    }


def _is_secret_path(value: str) -> bool:
    normalized = value.strip().replace("\\", "/").rstrip("/")
    name = PurePosixPath(normalized).name
    if not name or _SECRET_EXAMPLE.fullmatch(name):
        return False
    return bool(_SECRET_NAME.fullmatch(name))


def _paths(value: Any) -> tuple[list[str], bool]:
    found: list[str] = []
    valid = True

    def visit(item: Any, path_value: bool = False) -> None:
        nonlocal valid
        if isinstance(item, str):
            if path_value:
                if item.strip():
                    found.append(item)
                else:
                    valid = False
        elif isinstance(item, dict):
            has_path = not path_value
            for child_key, child in item.items():
                if not isinstance(child_key, str):
                    valid = False
                    continue
                is_path = child_key.lower() in _FILE_PATH_KEYS
                has_path = has_path or is_path
                visit(child, is_path)
            if not has_path:
                valid = False
        elif isinstance(item, list):
            if path_value and not item:
                valid = False
            for child in item:
                visit(child, path_value)
        elif path_value:
            valid = False

    visit(value)
    return found, valid


def _patch_paths(patch: str) -> list[str]:
    found = []
    for match in _PATCH_PATH.finditer(patch):
        path = match.group(1) or match.group(2)
        if path and path != "/dev/null":
            found.append(path)
    return found


def _is_freeform_patch(patch: str) -> bool:
    lines = patch.strip().splitlines()
    if not lines:
        return False
    markers = {"*** begin patch", "*** end patch"}
    if any(line.strip().lower() in markers for line in lines):
        if (
            lines[0].strip().lower() != "*** begin patch"
            or lines[-1].strip().lower() != "*** end patch"
            or sum(line.strip().lower() in markers for line in lines) != 2
        ):
            return False
        headers = list(_PATCH_HEADER.finditer(patch))
        if not headers:
            return False
        end_marker = re.search(r"(?im)^[ \t]*\*\*\* end patch[ \t]*\r?$", patch)
        if not end_marker or patch[patch.find("\n") + 1 : headers[0].start()].strip():
            return False
        for index, header in enumerate(headers):
            end = headers[index + 1].start() if index + 1 < len(headers) else end_marker.start()
            body = [line for line in patch[header.end() : end].splitlines() if line]
            action = header.group(1).lower()
            if action == "add" and any(not line.startswith("+") for line in body):
                return False
            if action == "delete" and any(line.strip() for line in body):
                return False
            if action == "update":
                if any(
                    not line.startswith((" ", "-", "+", "@@", "*** Move to:", "*** End of File"))
                    for line in body
                ):
                    return False
                if not any(line.startswith(("+", "-", "*** Move to:")) for line in body):
                    return False
        return True
    if _DELETE_PATCH.search(patch):
        return True
    if not re.search(r"(?m)^-(?!--)", patch) or not re.search(r"(?m)^\+(?!\+\+)", patch):
        return False
    return all(
        not line
        or line.startswith((" ", "-", "+", "@@", "*** Update File:", "*** Add File:", "*** Move to:", "\\"))
        for line in lines
    )


def _git_force_command(command: str) -> bool:
    tokens = re.findall(r'"[^"]*"|\'[^\']*\'|\S+', command.lower())
    options_with_values = {
        "-c",
        "-C",
        "--git-dir",
        "--work-tree",
        "--namespace",
        "--exec-path",
        "--config-env",
    }
    for index, token in enumerate(tokens):
        if token.strip("\"'") != "git":
            continue
        args = [item.strip("\"'") for item in tokens[index + 1 :]]
        cursor = 0
        while cursor < len(args) and args[cursor].startswith("-"):
            option = args[cursor]
            if option in options_with_values:
                cursor += 2
            elif any(option.startswith(f"{name}=") for name in options_with_values if name.startswith("--")):
                cursor += 1
            else:
                cursor += 1
        if cursor >= len(args):
            continue
        subcommand = args[cursor]
        subargs = args[cursor + 1 :]
        if subcommand == "reset" and "--hard" in subargs:
            return True
        if subcommand == "clean" and any(
            item == "--force" or (item.startswith("-") and not item.startswith("--") and "f" in item[1:])
            for item in subargs
        ):
            return True
        if subcommand == "push" and any(
            item in {"-f", "--force", "--force-with-lease"} for item in subargs
        ):
            return True
    return False


def evaluate(event: Any) -> dict[str, Any] | None:
    if not isinstance(event, dict):
        return _decision("ask", "Hookの入力がJSONオブジェクトではありません。操作内容を確認してから許可してください。")

    tool_name = event.get("toolName", event.get("tool_name", event.get("name", "")))
    tool_input = event.get(
        "toolArgs",
        event.get("tool_args", event.get("tool_input", event.get("toolInput", event.get("arguments")))),
    )
    if not isinstance(tool_name, str) or not tool_name.strip():
        return _decision("ask", "ツール情報が不足しています。操作内容を確認してから許可してください。")

    lowered_name = tool_name.strip().lower()
    if isinstance(tool_input, str):
        try:
            parsed_input = json.loads(tool_input)
        except json.JSONDecodeError:
            parsed_input = tool_input
        except (ValueError, RecursionError):
            return _decision("ask", "ツール情報を解析できません。操作内容を確認してから許可してください。")
        if (
            lowered_name in {"apply_patch", "applypatch", "edit"}
            and isinstance(parsed_input, str)
            and _is_freeform_patch(parsed_input)
        ):
            tool_input = {"patch": parsed_input}
        else:
            tool_input = parsed_input
    if not isinstance(tool_input, dict):
        return _decision("ask", "ツール情報が不足しています。操作内容を確認してから許可してください。")

    try:
        paths, valid_paths = _paths(tool_input)
    except RecursionError:
        return _decision("ask", "ツール情報を解析できません。操作内容を確認してから許可してください。")
    patch_text = " ".join(
        str(value)
        for key, value in tool_input.items()
        if isinstance(key, str)
        and isinstance(value, str)
        and key.lower() in _PATCH_TEXT_KEYS
    )
    paths.extend(_patch_paths(patch_text))
    if any(_is_secret_path(path) for path in paths):
        return _decision("deny", ".envの秘密情報を含むファイルへのエージェントツールからの直接アクセスは禁止されています。.env.exampleは対象外です。")
    if not valid_paths:
        return _decision("ask", "ファイル操作の対象を解析できません。内容を確認してから許可してください。")

    is_search = lowered_name in _SEARCH_TOOLS
    is_file = lowered_name in _FILE_TOOLS
    if is_file and not is_search and not paths:
        if lowered_name not in {"apply_patch", "applypatch", "patch", "edit"} or not _is_freeform_patch(patch_text):
            return _decision("ask", "ファイル操作の対象を特定できません。内容を確認してから許可してください。")

    try:
        text = json.dumps(
            {
                key: value
                for key, value in tool_input.items()
                if not (isinstance(value, str) and key.lower() in _PATCH_TEXT_KEYS)
            },
            ensure_ascii=False,
        )
    except (TypeError, ValueError, RecursionError):
        return _decision("ask", "ツール情報を解析できません。操作内容を確認してから許可してください。")
    secret_refs = [
        match.group(0)
        for match in _SECRET_REFERENCE.finditer(text)
        if not _SECRET_EXAMPLE.fullmatch(match.group(0))
    ]
    if secret_refs and (
        lowered_name in {"view", "create"}
        or any(word in lowered_name for word in ("read", "edit", "write", "delete", "file"))
    ):
        return _decision("deny", ".envの秘密情報を含むファイルへのエージェントツールからの直接アクセスは禁止されています。.env.exampleは対象外です。")

    if any(word in lowered_name for word in ("delete", "remove", "unlink")):
        return _decision("ask", "ファイル削除には、対象と理由を明示したユーザーの確認が必要です。")

    if _DELETE_PATCH.search(patch_text):
        return _decision("ask", "パッチにファイル削除が含まれています。実行前に対象と理由をユーザーに確認してください。")

    command = tool_input.get("command")
    is_known_shell = lowered_name in _SHELL_TOOLS
    if is_known_shell or any(word in lowered_name for word in ("terminal", "shell", "execute", "command")):
        if not isinstance(command, str) or not command.strip():
            return _decision("ask", "コマンドが指定されていないか、内容を解析できません。実行前に確認してください。")
        if _git_force_command(command):
            return _decision("deny", "Gitの履歴や作業内容を破壊する強制操作は禁止されています。ユーザーが承認していても実行できません。")
        command_secret_refs = [
            match.group(0)
            for match in _SECRET_REFERENCE.finditer(command)
            if not _SECRET_EXAMPLE.fullmatch(match.group(0))
        ]
        if command_secret_refs:
            if _SECRET_EXFIL.search(command):
                return _decision("deny", "シェルコマンドから.envの秘密情報を含むファイルや内容を外部サービスへ送信することは禁止されています。")
            if _SECRET_WRITE.search(command):
                return _decision("deny", "シェルコマンドによる.envの秘密情報を含むファイルの読み取り・編集・上書き・削除は禁止されています。")
            if _SECRET_READ.search(command):
                return _decision("deny", "シェルコマンドによる.envの秘密情報を含むファイルの表示・調査は禁止されています。")
            return _decision("ask", "コマンドが秘密情報を含むファイルを参照しています。秘密値を表示せずに目的を確認してください。")
        if _ENV_DUMP.search(command):
            return _decision("ask", "環境変数の一括表示は秘密情報を漏えいさせるおそれがあります。秘密値を含まない範囲の確認かどうか確認してください。")
        if _FILE_DELETE.search(command):
            return _decision("ask", "シェルからのファイル削除には、正確な対象と理由を明示したユーザーの確認が必要です。")
        if is_known_shell:
            return None
        return _decision("ask", "ツールの種類を判定できません。操作内容を確認してから許可してください。")

    if is_search:
        if not any(
            isinstance(value, str) and value.strip()
            for key, value in tool_input.items()
            if key.lower() in {"pattern", "query", "regex", "search", "searchterm", "search_term", "glob", "glob_pattern"}
        ):
            return _decision("ask", "検索条件を特定できません。内容を確認してから許可してください。")
        return None
    if is_file:
        return None
    return _decision("ask", "ツールの種類を判定できません。操作内容を確認してから許可してください。")


def main() -> int:
    try:
        event = json.load(getattr(sys.stdin, "buffer", sys.stdin))
    except (ValueError, UnicodeDecodeError, RecursionError):
        result = _decision("ask", "Hookへの入力が正しいJSONではありません。操作内容を確認してから許可してください。")
    else:
        result = evaluate(event)
    if result is not None:
        print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
