import logging
import sys

from .agent_client import AgentClient
from .audio_io import AudioRecorder
from .config import Config
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
    agent.register_handlers(make_handlers(robot))

    stt = STTService(config)
    tts = TTSService(config)
    recorder = AudioRecorder(config)

    try:
        run_loop(recorder, stt, agent, tts)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
