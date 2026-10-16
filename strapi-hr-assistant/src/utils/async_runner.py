import asyncio
import threading
from typing import Any, Coroutine


class _AsyncRunner:
    """Manages a dedicated background thread running a persistent asyncio event loop.
    
    This avoids 'RuntimeError: Event loop is closed' in Streamlit when caching
    async clients/models (like ChatGoogleGenerativeAI or MultiServerMCPClient)
    across multiple script reruns.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="AsyncRunnerThread")
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()


_runner = _AsyncRunner()


def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run an async coroutine synchronously on the persistent background event loop."""
    return _runner.run(coro)
