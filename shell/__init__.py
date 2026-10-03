"""Reachy realtime shell.

Robot embodiment/orchestration layer for the realtime voice rebuild
(docs/epic-realtime-rebuild.md). Owns the physical audio devices, attention
state, robot tools, and the OpenAI-Realtime client. The conversation
pipeline itself (VAD/turn/STT/LLM/TTS/cancellation) lives in the pinned
huggingface/speech-to-speech service; this package never reimplements it.
"""
