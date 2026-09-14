from __future__ import annotations

import os
import pytest
from app.core.config import PixelConfig, load_config
from app.core.errors import ConfigError

@pytest.mark.unit
def test_default_config_loads():
    config = PixelConfig()
    assert config is not None

@pytest.mark.unit
def test_config_values():
    config = PixelConfig()
    assert config.logging is not None
    assert config.security is not None

@pytest.mark.unit
def test_load_config_no_file():
    config = load_config(None)
    assert isinstance(config, PixelConfig)

@pytest.mark.unit
def test_env_override():
    os.environ['PIXEL_LOGGING__LEVEL'] = 'DEBUG'
    try:
        config = load_config(None)
        assert config.logging.level == 'DEBUG'
    finally:
        del os.environ['PIXEL_LOGGING__LEVEL']

@pytest.mark.unit
def test_invalid_config_file_path():
    with pytest.raises(ConfigError):
        load_config('nonexistent.toml')
