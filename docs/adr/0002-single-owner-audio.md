# ADR 0002: One persistent microphone owner, one speaker owner, PCM fan-out

- Status: accepted
- Date: 2026-10-02

## Context

The legacy app touched audio devices in at least four independent places: a fresh
`sd.InputStream` per turn (`audio_io.py`), a second InputStream inside
`InterruptiblePlayer` to watch for barge-in (plus a third capture of the rest of the
interruption), `sd.rec()` every 10 s in the mic watchdog (which also sampled the
*system default* device, not the Reachy one, and stomped the shared
`sounddevice._last_callback` via bare `sd.wait()`), and scattered `sd.play()` calls
that raced each other (the peekaboo doop-doop mistiming was exactly this). Every
clipped start, phantom barge-in, and device-contention race came from this design.

The 2026-07 spike (spike/s2s_client.py, commit b1de612) proved the alternative on
this hardware: one `InputStream` and one `OutputStream` opened at startup, never
reopened, everything else communicating through queues. 7498 frames sent, zero
errors; barge-in needed no local VAD because the server's `speech_started` event is
the trigger and queued playback is flushed on the floor.

## Decision

Exactly one component opens capture and one owns playback, for the lifetime of the
process. The mic owner publishes PCM to subscribers (KWS, ambient STT, the realtime
client while a lease is active, speaker/ASD experiments, metrics) via a fan-out ring
buffer that preserves pre-roll. The speaker owner is a mixer: realtime TTS audio,
prerendered acknowledgements, and earcons are channels with generation tags; a
cancellation drops stale generations before they become audible. Watchdogs observe
the owners' health signals; they never open a stream.

## Consequences

- A mute must be a first-class state on the mic owner (real mute, not a flag file),
  and watchdog logic must distinguish muted / silent-room / dead-stream /
  missing-device (the 2026-07-27 incident: PipeWire mute read as all-zero PCM and the
  watchdog rebooted a healthy XMOS ~60 times in 3 hours).
- Cancellation invariant: on interruption, flush queued output immediately, never
  "finish the sentence anyway", and keep the interrupting utterance with pre-roll.
- Experiment harnesses subscribe or don't run; they never touch the device.
