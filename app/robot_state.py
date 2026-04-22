import logging

from reachy_mini import ReachyMini

from .config import Config

log = logging.getLogger(__name__)


class RobotConnection:
    def __init__(self, config: Config):
        self.config = config
        self.mini: ReachyMini | None = None

    def connect(self) -> ReachyMini:
        log.info("Connecting to Reachy Mini at %s", self.config.reachy_host)
        self.mini = ReachyMini()
        self.mini.__enter__()
        log.info("Connected to Reachy Mini")
        return self.mini

    def disconnect(self):
        if self.mini:
            log.info("Disconnecting from Reachy Mini")
            self.mini.__exit__(None, None, None)
            self.mini = None

    @property
    def connected(self) -> bool:
        return self.mini is not None
