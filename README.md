# Pixel v2 — Intelligent Local AI Runtime

A personal AI assistant for Windows, designed as an intelligent local runtime that uses AI models as replaceable intelligence providers.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run Pixel
python main.py

# Run with config file
python main.py -c config/default.toml

# Run with debug logging
python main.py --log-level DEBUG

# Run tests
pytest tests/ -v
```

## Architecture

Pixel v2 is built around a central `PixelRuntime` that coordinates independent subsystems:

- **EventBus** — Typed async event system for decoupled communication
- **StateMachine** — Explicit state transitions (IDLE → UNDERSTANDING → THINKING → ...)
- **AI Providers** — Pluggable AI backends (Cloud, Ollama, Fallback)
- **Intent Router** — Deterministic fast-path for common commands + AI routing
- **Action Engine** — Safe, permission-checked action execution
- **Security Layer** — Centralized risk assessment and permission policies

## Design Principles

1. **Fast First** — Simple commands (time, calc, greetings) resolve in milliseconds without AI
2. **Model Independence** — AI providers are replaceable; the runtime owns execution
3. **Online + Local Hybrid** — Supports cloud AI, local models, and deterministic fallback
4. **Secure by Design** — AI never directly executes OS commands; all actions go through permission checks
5. **Observable** — Structured logging with request IDs and telemetry

## Project Status

**Phase 0** — Foundation ✓

- [x] Runtime lifecycle
- [x] Event bus
- [x] State machine
- [x] AI provider interface
- [x] Intent router (deterministic)
- [x] Action engine
- [x] Security layer
- [x] Configuration
- [x] Structured logging
- [x] Test suite
