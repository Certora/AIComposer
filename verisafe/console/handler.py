from typing import Any
import signal

class BrokenSignalHandler(Exception):
    def __init__(self):
        pass

class DebugHandler():
    def __init__(self):
        self.requested = False
        signal.signal(signal.SIGINT, self.catch)

    def catch(self, signum: int, stk: Any | None):
        if self.requested:
            raise BrokenSignalHandler()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("Debug console requested. (Control+C again to kill)")
        self.requested = True

    def reset(self):
        if not self.requested:
            raise BrokenSignalHandler()
        self.requested = False
        signal.signal(signal.SIGINT, self.catch)
