# Deploy

Production process layout (systemd user services, per the EPIC). Units live
here so they are versioned; install with:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llama-server reachy-vision reachy-llm-proxy reachy-s2s reachy-shell
```

Services, in startup order:

| Unit | Runs | Notes |
|---|---|---|
| `reachy-mini-daemon` | robot hardware | unchanged; units live outside repo until Phase 2 |
| `llama-server` | chat LLM on :8080 (Granite 4.1 3B, text-only) | `scripts/run_llama_server.sh`; see issue #33 for pinning |
| `reachy-vision` | camera VLM on :8082, CPU (Qwen3.5-0.8B) | `scripts/run_vision_server.sh`; set `VISION_BASE_URL=http://127.0.0.1:8082/v1` in `.env` |
| `reachy-llm-proxy` | `python -m shell.llm_proxy` on :8081 | rewrites STT mis-hearings of "Reachy" before llama.cpp |
| `reachy-s2s` | pinned HF speech-to-speech realtime | `scripts/install_s2s.sh` once, then this |
| `reachy-shell` | audio owners + attention + tools (`python -m shell`) | the new stack |

Phase 2 (issue #22) will fold the legacy `reachy-agent` / `qwen3-tts` units
away and teach the dashboard about these names.

## Prereqs on a fresh machine

```bash
scripts/install_s2s.sh        # clones + pins the service in ~/.local/share/reachy
uv sync --extra kws
scripts/get_kws_model.sh      # wake-word model + tokenized keywords into models/kws
scripts/build_qwentts.sh      # only for S2S_TTS=qwen3: Pascal build of the Qwen3-TTS engine
HF_HUB_OFFLINE=0 scripts/run_s2s.sh   # once, to download Parakeet/Kokoro; Ctrl-C when "Uvicorn running"
```
