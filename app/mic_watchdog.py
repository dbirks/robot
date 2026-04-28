"""XMOS XVF3800 microphone watchdog.

Monitors the mic capture stream and sends an XMOS REBOOT command
when the mic goes silent (all-zero audio). The XMOS chip re-enumerates
on USB and audio capture resumes within ~5 seconds.

Background: The XVF3800 firmware has a known bug where the USB audio
capture endpoint stops sending data while playback and DOA continue
to work. USB autosuspend can trigger this. The REBOOT command via
vendor control transfer is the software equivalent of a USB replug.
"""

import logging
import threading
import time

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

CHECK_INTERVAL = 10.0
SILENCE_THRESHOLD = 5
CONSECUTIVE_FAILURES = 3
RECOVERY_WAIT = 8.0


class MicWatchdog:
    def __init__(self, sample_rate: int = 16000):
        self._sr = sample_rate
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._consecutive_silent = 0

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._watch_loop, name="mic-watchdog", daemon=True)
        self._thread.start()
        log.info("MicWatchdog started (check every %.0fs, recover after %d silent checks)", CHECK_INTERVAL, CONSECUTIVE_FAILURES)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        log.info("MicWatchdog stopped")

    def _check_mic(self) -> bool:
        try:
            audio = sd.rec(int(self._sr * 0.5), samplerate=self._sr, channels=1, dtype="int16")
            sd.wait()
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            return rms > SILENCE_THRESHOLD
        except Exception:
            return False

    def _reboot_xmos(self):
        try:
            from reachy_mini.media.audio_control_utils import init_respeaker_usb

            respeaker = init_respeaker_usb()
            log.warning("XMOS mic silent — sending REBOOT command")
            respeaker.write("REBOOT", [1])
            log.info("XMOS REBOOT sent, waiting %.0fs for re-enumeration...", RECOVERY_WAIT)
            time.sleep(RECOVERY_WAIT)
            if self._check_mic():
                log.info("XMOS mic recovered!")
            else:
                log.error("XMOS mic still silent after REBOOT")
        except Exception:
            log.exception("Failed to reboot XMOS")

    def _watch_loop(self):
        time.sleep(CHECK_INTERVAL)
        while not self._stop.is_set():
            if self._check_mic():
                self._consecutive_silent = 0
            else:
                self._consecutive_silent += 1
                log.debug("Mic silent (%d/%d)", self._consecutive_silent, CONSECUTIVE_FAILURES)
                if self._consecutive_silent >= CONSECUTIVE_FAILURES:
                    self._reboot_xmos()
                    self._consecutive_silent = 0
            self._stop.wait(CHECK_INTERVAL)
