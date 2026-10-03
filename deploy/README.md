# Deploy

Production process layout (systemd user services, per the EPIC). Units live
here so they are versioned; install with:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now reachy-s2s reachy-shell
```

Services, in startup order:

| Unit | Runs | Notes |
|---|---|---|
| `reachy-mini-daemon` | robot hardware | unchanged; units live outside repo until Phase 2 |
| `llama-server` | LLM inference | `scripts/run_llama_server.sh`; see issue #33 for pinning |
| `reachy-s2s` | pinned HF speech-to-speech realtime | `scripts/install_s2s.sh` once, then this |
| `reachy-shell` | audio owners + attention + tools (`python -m shell`) | the new stack |

Phase 2 (issue #22) will fold the legacy `reachy-agent` / `qwen3-tts` units
away and teach the dashboard about these names.

## Prereqs on a fresh machine

```bash
scripts/install_s2s.sh        # clones + pins the service in ~/.local/share/reachy
uv sync --extra kws           # KWS model download is tracked in issue #23 (not yet pinned)
```
