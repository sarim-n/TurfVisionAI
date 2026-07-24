"""
Unit tests for structured JSON logger.
"""

import json
import logging
from shared.logging import JSONFormatter, setup_logger


def test_json_formatter_structure():
    formatter = JSONFormatter(service_name="test_service")
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test log message",
        args=(),
        exc_info=None,
    )
    output = formatter.format(record)
    data = json.loads(output)

    assert data["service"] == "test_service"
    assert data["level"] == "INFO"
    assert data["message"] == "Test log message"
    assert "timestamp" in data


def test_setup_logger():
    logger = setup_logger(name="unit_test_logger", service_name="vision_service")
    assert logger.name == "unit_test_logger"
    assert len(logger.handlers) >= 1
