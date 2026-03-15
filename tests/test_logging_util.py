"""Unit tests for logging_util.py — logging setup."""

import os
import logging


class TestSetupLogging:
    def test_creates_log_directory(self, tmp_path):
        from logging_util import setup_logging
        log_dir = str(tmp_path / "test_logs")
        logger = setup_logging(log_dir=log_dir)
        assert os.path.isdir(log_dir)
        # Cleanup handlers to avoid file locks
        for h in logger.handlers[:]:
            h.close()
            logger.removeHandler(h)

    def test_creates_log_file(self, tmp_path):
        from logging_util import setup_logging
        log_dir = str(tmp_path / "test_logs")
        logger = setup_logging(log_dir=log_dir)
        log_files = os.listdir(log_dir)
        assert len(log_files) == 1
        assert log_files[0].startswith("app_")
        assert log_files[0].endswith(".log")
        for h in logger.handlers[:]:
            h.close()
            logger.removeHandler(h)

    def test_logger_has_handlers(self, tmp_path):
        from logging_util import setup_logging
        log_dir = str(tmp_path / "test_logs")
        logger = setup_logging(log_dir=log_dir)
        handler_types = [type(h) for h in logger.handlers]
        assert logging.StreamHandler in handler_types
        assert logging.FileHandler in handler_types
        for h in logger.handlers[:]:
            h.close()
            logger.removeHandler(h)

    def test_logger_name(self, tmp_path):
        from logging_util import setup_logging
        log_dir = str(tmp_path / "test_logs")
        logger = setup_logging(log_dir=log_dir)
        assert logger.name == "cauldron"
        for h in logger.handlers[:]:
            h.close()
            logger.removeHandler(h)
