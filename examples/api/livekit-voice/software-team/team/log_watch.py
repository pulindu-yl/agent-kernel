"""Reading a line another library logs, as the example's shortcut for an event it does not otherwise get."""

import logging
from typing import Any, ClassVar

_log = logging.getLogger("software_team.log_watch")


class LogLineWatcher(logging.Handler):
    """A logging handler that reacts to lines from the loggers in ``LOGGERS``.

    Three parts of the example learn of something only from a log line: the caller's words
    (``Caller said: ...``), each agent turn the gateway played (``AI response: ...``) and the model
    client's rate-limit retries. That depends on the wording of those lines and on the loggers
    emitting INFO, so ``install`` warns when a logger is set above INFO, which would silence the
    watcher without an error.
    """

    LOGGERS: ClassVar[tuple[str, ...]] = ()

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)

    @classmethod
    def install(cls, *args: Any) -> "LogLineWatcher":
        """Create the watcher with ``args`` and attach it to each of its loggers."""
        watcher = cls(*args)
        for name in cls.LOGGERS:
            logger = logging.getLogger(name)
            logger.addHandler(watcher)
            if not logger.isEnabledFor(logging.INFO):
                _log.warning(
                    f"{cls.__name__} reads INFO lines from '{name}', which is set to log less; it will see nothing"
                )
        return watcher

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.line(record.getMessage())
        except Exception:
            # Never let reading a log line break the code that logged it.
            self.handleError(record)

    def line(self, message: str) -> None:
        """React to one line from a watched logger."""
        raise NotImplementedError
