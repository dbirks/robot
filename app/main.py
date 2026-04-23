import logging
import sys
import threading

from .agent_client import AgentClient
from .audio_io import AudioRecorder
from .config import Config
from .face_tracker import FaceTracker
from .head_wobbler import HeadWobbler
from .movement_manager import MovementManager
from .orchestrator import run_loop
from .robot_state import RobotConnection
from .robot_tools import TOOLS, make_handlers
from .stt_service import STTService
from .tts_service import TTSService


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

    agent = AgentClient(config, tools=TOOLS)
    agent.register_handlers(make_handlers(robot, agent=agent))

    stt = STTService(config)
    tts = TTSService(config)
    recorder = AudioRecorder(config)

    movement = None
    wobbler = None
    tracker = None
    stop_event = threading.Event()

    if robot.connected and robot.mini is not None:
        movement = MovementManager(robot.mini)
        wobbler = HeadWobbler(movement.set_speech_offsets)
        movement.start()
        wobbler.start()

        tracker = FaceTracker()
        tracker.start_tracking(robot.mini, movement, stop_event)

    try:
        run_loop(recorder, stt, agent, tts, movement, wobbler)
    finally:
        stop_event.set()
        if tracker:
            tracker.stop_tracking()
        if wobbler:
            wobbler.stop()
        if movement:
            movement.stop()
        robot.disconnect()


if __name__ == "__main__":
    main()
