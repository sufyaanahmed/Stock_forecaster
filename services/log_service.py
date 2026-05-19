"""
Centralized Log Service
=======================
Collects structured log records from all subsystems and exposes them
via an in-memory ring-buffer for SSE/polling by the frontend.

Log record schema:
  {ts, level, source, message}

Levels: INFO | SUCCESS | WARNING | ERROR
Sources: api | training | backtest | features | cache | macro
"""

import logging
import time
from collections import deque
from typing import List, Dict, Any

# ── Ring buffer ───────────────────────────────────────────────────────────────
_LOG_BUFFER: deque = deque(maxlen=500)   # keep last 500 records

_LEVEL_MAP = {
    logging.DEBUG:    "INFO",
    logging.INFO:     "INFO",
    logging.WARNING:  "WARNING",
    logging.ERROR:    "ERROR",
    logging.CRITICAL: "ERROR",
}


def _source_from_name(name: str) -> str:
    """Derive a short source label from the Python logger name."""
    if "train" in name:     return "training"
    if "backtest" in name:  return "backtest"
    if "feature" in name:   return "features"
    if "market" in name:    return "market"
    if "macro" in name:     return "macro"
    if "cache" in name:     return "cache"
    if "rank" in name:      return "ranking"
    return "api"


class FrontendLogHandler(logging.Handler):
    """Python logging handler that feeds the ring buffer."""

    def emit(self, record: logging.LogRecord):
        try:
            level = _LEVEL_MAP.get(record.levelno, "INFO")
            # Mark certain messages as SUCCESS
            msg = self.format(record)
            if level == "INFO" and any(
                kw in msg.lower()
                for kw in ("done", "complete", "success", "saved", "loaded", "cached")
            ):
                level = "SUCCESS"

            _LOG_BUFFER.append({
                "ts":      round(time.time() * 1000),   # epoch ms
                "level":   level,
                "source":  _source_from_name(record.name),
                "message": msg,
            })
        except Exception:
            pass   # never crash the caller


def install_handler(logger_name: str = ""):
    """
    Attach FrontendLogHandler to a logger (default: root logger).
    Safe to call multiple times — only installs once.
    """
    target = logging.getLogger(logger_name)
    for h in target.handlers:
        if isinstance(h, FrontendLogHandler):
            return   # already installed
    handler = FrontendLogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    target.addHandler(handler)
    target.setLevel(logging.DEBUG)


def get_logs(since_ts: int = 0, limit: int = 200) -> List[Dict[str, Any]]:
    """
    Return log records newer than `since_ts` (epoch ms), up to `limit`.
    """
    records = [r for r in _LOG_BUFFER if r["ts"] > since_ts]
    return records[-limit:]


def push_log(level: str, source: str, message: str):
    """
    Push a log record directly (use for structured internal events).
    """
    _LOG_BUFFER.append({
        "ts":      round(time.time() * 1000),
        "level":   level.upper(),
        "source":  source,
        "message": message,
    })
