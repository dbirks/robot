# Reachy Mini + OpenAI Realtime API

Voice-controlled Reachy Mini robot using OpenAI's Realtime API and Agents SDK. The Realtime API streams voice input/output and enables the model to call Python functions (tools) to control the robot. We use the [reachy-mini SDK](https://github.com/pollen-robotics/reachy_mini) for robot control.

Supports both OpenAI (`gpt-4o-realtime` or `gpt-4o-mini-realtime`) and Azure OpenAI.

## Quick Start

```bash
# Terminal 1: Start simulator
reachy-mini-daemon --sim

# Terminal 2: Set up and run agent
cp .env.example .env
# Edit .env with your OpenAI API key
uv sync
uv run python -m app.agent
```

## Project Structure

```
app/
    __init__.py
    tools.py      # Robot motion functions
    agent.py      # Realtime agent setup
main.py           # Original example (not used)
pyproject.toml    # Dependencies
.env.example      # API key template
```
