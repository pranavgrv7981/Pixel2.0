from __future__ import annotations

import pytest
from app.core import logging
from app.core.logging import setup_logging, get_logger, bind_request_context

@pytest.fixture(autouse=True)
def reset_logging():
    logging._configured = False
    yield
    logging._configured = False

@pytest.mark.unit
def test_setup_logging_does_not_error():
    setup_logging()

@pytest.mark.unit
def test_get_logger():
    setup_logging()
    logger = get_logger('test')
    assert logger is not None

@pytest.mark.unit
def test_bind_request_context():
    setup_logging()
    bind_request_context('req-123')
