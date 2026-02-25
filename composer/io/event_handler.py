from abc import ABC, abstractmethod


class EventHandler(ABC):
    @abstractmethod
    async def handle_event(self, payload: dict, path: list[str], checkpoint_id: str) -> None: ...


class NullEventHandler(EventHandler):
    async def handle_event(self, payload: dict, path: list[str], checkpoint_id: str) -> None:
        pass
