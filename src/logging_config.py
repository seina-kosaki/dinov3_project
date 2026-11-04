"""ロギング初期化モジュール。

コンソール（stderr）とプロジェクト直下 ``logs/YYYY-MM-DD.log`` の両方へ
ログを出力するようルートロガーを設定する。詳細は ``docs/logger-spec.md`` を参照。
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import TextIO

_LOG_DIRECTORY: Path = Path(__file__).resolve().parent.parent / "logs"

# 本モジュールがルートロガーに登録したハンドラー。再初期化時にこれらのみを置き換える。
_handlers: tuple[logging.Handler, ...] = ()


class _DailyFileHandler(logging.Handler):
    """レコードの日付（ローカル時刻）ごとに ``YYYY-MM-DD.log`` へ追記するハンドラー。

    ファイルの作成・書き込み・日付切替の失敗は呼び出し元へ例外として伝播する。
    """

    def __init__(self, directory: Path) -> None:
        """ログディレクトリを作成し、当日のログファイルを開く。

        Args:
            directory: 日別ログファイルの保存先ディレクトリ。
        """
        super().__init__()
        self._directory = directory
        self._date: date | None = None
        self._stream: TextIO | None = None
        directory.mkdir(parents=True, exist_ok=True)
        self._open_for_date(datetime.now().date())

    def _open_for_date(self, day: date) -> TextIO:
        """現在のファイルを閉じ、指定日のログファイルを追記モードで開く。

        Args:
            day: 開くログファイルの日付。

        Returns:
            開いたログファイルのストリーム。
        """
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        self._stream = (self._directory / f"{day.isoformat()}.log").open("a", encoding="utf-8")
        self._date = day
        return self._stream

    def emit(self, record: logging.LogRecord) -> None:
        """レコードの日付に対応するファイルへ1行書き込み、即座にフラッシュする。"""
        day = datetime.fromtimestamp(record.created).date()
        stream = self._stream
        if stream is None or day != self._date:
            stream = self._open_for_date(day)
        stream.write(self.format(record) + "\n")
        stream.flush()

    def close(self) -> None:
        """ログファイルを閉じ、ハンドラーを終了する。"""
        self.acquire()
        try:
            if self._stream is not None:
                try:
                    self._stream.close()
                finally:
                    self._stream = None
        finally:
            self.release()
            super().close()


def setup_logging(level: int | str = logging.INFO) -> None:
    """ルートロガーをコンソールと日別ログファイルへ出力するよう初期化する。

    再呼び出し時は本モジュールが登録したハンドラーのみを置き換え、
    他のハンドラーには触れないため、ログが重複出力されない。

    Args:
        level: ログレベル。``logging`` が受け付ける整数またはレベル名
            （例: ``logging.DEBUG``、``"DEBUG"``）。

    Raises:
        TypeError: ``level`` が整数でもレベル名でもない場合。
        ValueError: ``level`` が未知のレベル名の場合。
        OSError: ログディレクトリの作成やログファイルのオープンに失敗した場合。
    """
    global _handlers

    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console.setFormatter(formatter)

    file_handler = _DailyFileHandler(_LOG_DIRECTORY)
    file_handler.setLevel(console.level)
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    previous_handlers = _handlers
    for handler in previous_handlers:
        root.removeHandler(handler)
    root.setLevel(console.level)
    root.addHandler(console)
    root.addHandler(file_handler)
    _handlers = (console, file_handler)
    for handler in previous_handlers:
        handler.close()
