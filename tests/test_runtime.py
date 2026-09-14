from __future__ import annotations

import pytest
from app.runtime.runtime import PixelRuntime
from app.core.config import PixelConfig
from app.core.types import PixelStateEnum

@pytest.fixture
def runtime():
    config = PixelConfig()
    return PixelRuntime(config)

@pytest.mark.unit
@pytest.mark.asyncio
async def test_runtime_start(runtime):
    await runtime.start()
    assert runtime.state_machine.state == PixelStateEnum.IDLE

@pytest.mark.unit
@pytest.mark.asyncio
async def test_runtime_stop(runtime):
    await runtime.start()
    await runtime.stop()

@pytest.mark.unit
@pytest.mark.asyncio
async def test_runtime_start_stop_lifecycle(runtime):
    await runtime.start()
    assert runtime.state_machine.state == PixelStateEnum.IDLE
    await runtime.stop()

@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_time_input(runtime):
    await runtime.start()
    response = await runtime.handle_input("what time is it")
    assert response is not None
    assert isinstance(response, str)

@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_greeting(runtime):
    await runtime.start()
    response = await runtime.handle_input("hello")
    assert response is not None
    assert isinstance(response, str)

@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_unknown(runtime):
    await runtime.start()
    response = await runtime.handle_input("explain quantum physics")
    assert response is not None
    assert isinstance(response, str)
