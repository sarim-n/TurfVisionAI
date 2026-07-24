"""
Purpose: Centralized structured JSON logging configuration.
Dependencies: logging, json, datetime
Inputs: Module name, log level
Outputs: Standardized structured logger instance
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter producing structured log lines for microservices."""

    def __init__(self, service_name: str = "turfvision"):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        log_object: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": self.service_name,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "file": f"{record.filename}:{record.lineno}",
        }

        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)

        # Include custom extra fields if present
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            log_object.update(record.extra_fields)

        return json.dumps(log_object)


def setup_logger(name: str, service_name: str = "turfvision", level: str = "INFO") -> logging.Logger:
    """Creates and configures a structured JSON logger."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid adding duplicate handlers if logger is re-initialized
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter(service_name=service_name))
        logger.addHandler(handler)
        logger.propagate = False

    return logger
