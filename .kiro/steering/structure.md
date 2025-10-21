# Project Structure

## Directory Layout

```
reachy-robot/
├── app/                    # Main application module
│   ├── __init__.py        # Package initialization
│   ├── agent.py           # Realtime agent setup and main loop
│   └── tools.py           # Robot motion functions and control
├── main.py                # Original example (not used by agent)
├── pyproject.toml         # Project dependencies and metadata
├── .env.example           # Environment variable template
├── .env                   # Local environment variables (gitignored)
└── README.md              # Project documentation
```

## Module Organization

### `app/agent.py`
- **Purpose**: Main entry point for the realtime agent
- **Contains**: Agent configuration, audio I/O handling, function tool definitions
- **Key Components**:
  - Robot function tools (`@function_tool` decorators)
  - Audio processing loops (mic input, speaker output)
  - OpenAI Realtime API session management
  - Event handling for voice interaction

### `app/tools.py`
- **Purpose**: Robot control abstraction layer
- **Contains**: `Robot` class with motion methods
- **Key Methods**:
  - `nod()`, `shake()`: Head gestures
  - `look_at()`: Absolute positioning
  - `antenna_wiggle()`: Antenna animations
- **Pattern**: Context manager for connection handling

### Configuration Files
- **`.env`**: Contains `OPENAI_API_KEY` (required)
- **`pyproject.toml`**: Python 3.13 requirement, core dependencies
- **`.python-version`**: Python version specification

## Code Organization Patterns

### Function Tools
- Use `@function_tool` decorator for robot actions
- Keep tool functions async and lightweight
- Always use Robot context manager (`with ROBOT:`)
- Return descriptive strings for user feedback

### Robot Control
- All motion methods are async
- Parameters are clamped for safety (angles, durations, counts)
- Movements return to neutral position after gestures
- Connection reuse via context manager pattern

### Audio Processing
- 24kHz input for OpenAI API compatibility
- 48kHz output for system audio
- Queue-based threading for real-time processing
- Automatic resampling between rates