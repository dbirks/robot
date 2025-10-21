# Technology Stack

## Core Technologies

- **Python 3.13**: Required for scipy dependency compatibility
- **OpenAI Realtime API**: Voice interaction and AI agent capabilities
- **Reachy Mini SDK**: Robot hardware control and simulation
- **OpenAI Agents SDK**: Framework for building AI agents with function tools

## Key Dependencies

- `reachy-mini[mujoco]==1.0.0rc5`: Robot control and MuJoCo simulation
- `openai-agents==0.4.0`: OpenAI agent framework
- `python-dotenv==1.1.1`: Environment variable management
- `sounddevice`, `scipy`: Audio processing and resampling
- `numpy`: Numerical operations for audio and motion

## Build System

- **uv**: Primary package manager (recommended)
- **pip**: Alternative package manager
- Project uses `pyproject.toml` for dependency management

## Common Commands

### Setup
```bash
# Install dependencies (preferred)
uv sync

# Alternative installation
pip install -e .
```

### Development Workflow
```bash
# Terminal 1: Start robot simulator
reachy-mini-daemon --sim

# Terminal 2: Run the agent
python -m app.agent
```

### Hardware Usage
```bash
# USB connection
reachy-mini-daemon

# Wireless (on Raspberry Pi)
reachy-mini-daemon
```

### Testing
- Use `modalities: ["text"]` in agent.py for text-only testing
- Switch to `modalities: ["audio"]` for voice interaction
- Simulator available at `http://localhost:8000/docs`

## Architecture Pattern

The project follows a layered architecture:
1. **OpenAI Realtime API** → Voice/text input
2. **Agents SDK Runner** → Function tool orchestration  
3. **Robot Tools Layer** → Motion abstraction
4. **Reachy Mini SDK** → Hardware/simulation interface
5. **REST API Daemon** → Low-level robot control