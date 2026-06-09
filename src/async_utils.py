import asyncio
from contextlib import asynccontextmanager


class AsyncMutex[T]:
    def __init__(self, value: T):
        self._value = value
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def lock(self):
        async with self._lock:
            yield self._value


class AsyncDict[K, V]:
    def __init__(self):
        self._data: dict[K, V] = {}
        self._events: dict[K, asyncio.Event] = {}

    def set(self, key: K, value: V) -> None:
        self._data[key] = value

        event = self._events.get(key)
        if event is not None:
            event.set()

    async def pop(self, key: K) -> V:
        if key not in self._data:
            event = self._events.setdefault(key, asyncio.Event())
            await event.wait()

        value = self._data.pop(key)
        self._events.pop(key, None)

        return value
