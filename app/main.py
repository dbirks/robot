import logging
import sys
import threading
import time

import numpy as np

from .agent_client import AgentClient
from .audio_io import AudioRecorder
from .config import Config
from .face_tracker import FaceTracker
from .head_wobbler import HeadWobbler
from .movement_manager import MovementManager
from .orchestrator import run_loop
from .robot_state import RobotConnection
from .robot_tools import SLEEP_ANTENNAS, SLEEP_HEAD_POSE, TOOLS, make_handlers
from .stt_service import STTService
from .tts_service import TTSService

INIT_ANTENNAS = [-0.1745, 0.1745]


def main():
    config = Config()
    config.logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.logs_dir / "agent.log"),
        ],
    )
    log = logging.getLogger(__name__)
    log.info("Starting reachy-local-agent")

    robot = RobotConnection(config)
    try:
        robot.connect()
    except Exception:
        log.warning("Could not connect to Reachy Mini — running without robot")

    stt = STTService(config)
    tts = TTSService(config)
    recorder = AudioRecorder(config)

    movement = None
    wobbler = None
    tracker = None
    stop_event = threading.Event()

    if robot.connected and robot.mini is not None:
        log.info("Waking up robot...")
        robot.mini.goto_target(head=np.eye(4), antennas=INIT_ANTENNAS, duration=2)
        time.sleep(0.5)

        movement = MovementManager(robot.mini)
        wobbler = HeadWobbler(movement.set_speech_offsets)
        movement.start()
        wobbler.start()

        tracker = FaceTracker()
        tracker.start_tracking(robot.mini, movement, stop_event)

    sleep_event = threading.Event()

    agent = AgentClient(config, tools=TOOLS)
    agent.register_handlers(make_handlers(robot, agent=agent, face_tracker=tracker, sleep_event=sleep_event))

    try:
        run_loop(recorder, stt, agent, tts, movement, wobbler, sleep_event=sleep_event, robot_mini=robot.mini)
    finally:
        stop_event.set()
        if tracker:
            tracker.stop_tracking()
        if wobbler:
            wobbler.stop()
        if movement:
            movement.stop()
        if robot.mini is not None:
            try:
                log.info("Putting robot to sleep...")
                robot.mini.goto_target(head=SLEEP_HEAD_POSE, antennas=SLEEP_ANTENNAS, duration=2)
                time.sleep(2)
            except Exception:
                log.exception("Sleep animation failed")
        robot.disconnect()


if __name__ == "__main__":
    main()
