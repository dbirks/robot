# ADR 0005: XVF3800 mic array — verified hardware behavior and watchdog design

- Status: accepted
- Date: 2026-10-02
- Source: measured on the actual robot, preserved from beads robot-7e6/6wr/dqj/kds/04e
  and app/mic_watchdog.py; upstream issues reachy_mini #845, #820, #389, #1161

## Verified hardware facts

- **Silent-mic firmware bug is real and recoverable in software.** USB autosuspend
  suspends the capture endpoint while playback stays active; the XMOS firmware does
  not recover. Prevention: udev rule `/etc/udev/rules.d/99-reachy-mini.rules` with
  `ATTR{power/autosuspend}="-1"`. Recovery: the XMOS `REBOOT` vendor control transfer
  (`respeaker.write('REBOOT', [1])`) restarts the DSP; mic back in ~8 s, no replug.
- **The array is LINEAR** (`AEC_MIC_ARRAY_TYPE=1`, mics at x=-49.95/-16.65/+16.65/
  +49.95 mm). DOA azimuth spans 0..pi with front/back ambiguity — see ADR 0003.
- **AEC is on-chip and there is no server-side software AEC** in the s2s service.
  Whether the far-end reference is actually wired is UNVERIFIED on the Lite —
  reachy_mini#1161 shows the daemon only enabling AEC routing when the wireless
  .asoundrc is present. If the reference is not wired, every TTS utterance
  phantom-triggers barge-in and every threshold tuned afterwards is garbage.
  Verify FIRST, never stack software AEC blindly.
- **Unexploited controls** (XMOS command appendix): `AEC_FIXEDBEAMSONOFF` /
  `AEC_FIXEDBEAMSGATING` (point both fixed beams at the seating azimuth ±20°,
  silences off-axis sources before the pipeline; ~10 lines; does NOT help a TV
  directly front/behind because of the linear array). `AEC_SPENERGY_VALUES` (4
  per-beam energies — two simultaneous non-zero separated azimuths is a
  competing-source signal DOA cannot provide). `PP_AGCONOFF` — AGC currently boosts
  the TV during robot silences. Startup tuning stolen from the Pollen conversation
  app: `PP_AGCMAXGAIN=10.0 PP_MIN_NS=0.8 PP_MIN_NN=0.8 PP_GAMMA_E=0.5
  PP_GAMMA_ETAIL=0.5 PP_NLATTENONOFF=0`.

- **Now used (2026-10-10, `shell/audio/xmos_tuning.py`):** `PP_AGCMAXGAIN` capped
  at 10 (boot value read on the robot: 64; `REACHY_XMOS_AGC_MAX_GAIN`, 0 = AGC off),
  re-applied every 10 s because a REBOOT restores it. On a wake word both fixed
  beams are steered at the wake DOA (`AEC_FIXEDBEAMSONOFF=1`, gating off) and
  released within ~1 s of the lease ending, so KWS always hears the auto beam
  (`REACHY_XMOS_BEAM_FOCUS=0` disables). Verified: params read back, mic keeps
  streaming. Not yet measured: actual off-axis attenuation in the room.

## Watchdog design rules (from the 2026-07-27 reboot-loop incident)

1. Monitor the **mic owner's** health signal (ADR 0002); never open a competing
   stream (`sd.rec()` without `device=` sampled the wrong input; bare `sd.wait()`
   blocked on sounddevice's module-global `_last_callback` and could kill the shared
   playback stream).
2. Distinguish four states before declaring firmware death: explicitly muted (PipeWire
   source mute reads as all-zero PCM — checking `wpctl`/port state is mandatory),
   silent room, dead/stalled stream, missing USB device. A mute must never trigger
   REBOOT (observed failure: ~60 reboots in 3 h against healthy hardware).
3. Rate-limit identical failure log lines (they were a large fraction of a 794 MB log).
