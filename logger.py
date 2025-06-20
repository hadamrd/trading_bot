import datetime
import logging
import sys
from pathlib import Path

LOGS_PATH = Path(__file__).parent.parent / "logs"
if not LOGS_PATH.exists():
    LOGS_PATH.mkdir()

class Logger(logging.Logger):
    def __init__(self, name="Logger", consoleOut=False):
        super().__init__(name)
        self.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s.%(msecs)03d | %(levelname)s ] %(message)s", datefmt="%H:%M:%S")
        now = datetime.datetime.now()
        fileHandler = logging.FileHandler(LOGS_PATH / f"{name}_{now.strftime('%Y-%m-%d')}.log")
        fileHandler.setFormatter(formatter)
        self.addHandler(fileHandler)
        if consoleOut:
            streamHandler = logging.StreamHandler(sys.stdout)
            streamHandler.setFormatter(formatter)
            self.addHandler(streamHandler)