"""
logger_setup.py
Shared application logger. Writes every logged action to logs/YYYYMMDD.log,
switching to a new file automatically when the date rolls over (no restart
required). Also echoes to the console, which is useful when running
`python3 server.py` in the foreground.
"""
import logging
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "../logs")
os.makedirs(LOG_DIR, exist_ok=True)


class DailyFileHandler(logging.Handler):
    """Logging handler that writes to logs/YYYYMMDD.log based on the current
    local date, opening a new file the first time a record is emitted after
    midnight."""

    def __init__(self, log_dir=LOG_DIR):
        super().__init__()
        self.log_dir = log_dir
        self._current_date = None
        self._stream = None
        self._open_for_today()

    def _log_path_for(self, date_str):
        return os.path.join(self.log_dir, f"spam_detection_part2_{date_str}.log")

    def _open_for_today(self):
        today = datetime.now().strftime("%Y%m%d")
        if today == self._current_date and self._stream is not None:
            return
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
        self._current_date = today
        self._stream = open(self._log_path_for(today), "a", encoding="utf-8")

    def emit(self, record):
        try:
            self._open_for_today()
            msg = self.format(record)
            self._stream.write(msg + "\n")
            self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
        super().close()


_LOGGER_NAME = "spam_detection"
_configured = False


def get_logger():
    """Return the shared application logger, configuring it on first call."""
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)

    if not _configured:
        logger.setLevel(logging.INFO)
        logger.propagate = False

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = DailyFileHandler()
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

        _configured = True

    return logger
