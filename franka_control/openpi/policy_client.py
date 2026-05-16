"""OpenPI websocket policy client."""

from __future__ import annotations

import logging
import threading
from typing import Any

from franka_control.openpi import msgpack_numpy

logger = logging.getLogger(__name__)


class OpenPIPolicyClient:
    """Thin synchronous client for an OpenPI policy server."""

    def __init__(
        self,
        host: str,
        port: int,
        api_key: str | None = None,
        *,
        scheme: str = "ws",
    ) -> None:
        self._uri = f"{scheme}://{host}:{port}"
        self._api_key = api_key
        self._packer = msgpack_numpy.Packer()
        self._infer_lock = threading.Lock()
        self._ws = None
        self.metadata: dict[str, Any] | None = None
        self._connect()

    def _connect(self) -> None:
        try:
            import websockets.sync.client
        except ImportError as exc:
            raise ImportError(
                "websockets is required for OpenPI inference. "
                "Install with: pip install websockets"
            ) from exc

        headers = (
            {"Authorization": f"Api-Key {self._api_key}"}
            if self._api_key
            else None
        )
        logger.info("Connecting to OpenPI policy server at %s", self._uri)
        kwargs = {
            "compression": None,
            "max_size": None,
        }
        if headers:
            kwargs["additional_headers"] = headers
        try:
            self._ws = websockets.sync.client.connect(self._uri, **kwargs)
        except TypeError:
            if "additional_headers" not in kwargs:
                raise
            kwargs["extra_headers"] = kwargs.pop("additional_headers")
            self._ws = websockets.sync.client.connect(self._uri, **kwargs)
        self.metadata = msgpack_numpy.unpackb(self._ws.recv())
        logger.info("Connected to OpenPI policy server: %s", self.metadata)

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Send one observation and return the decoded policy response."""
        with self._infer_lock:
            if self._ws is None:
                self._connect()

            self._ws.send(self._packer.pack(observation))
            response = self._ws.recv()
            if isinstance(response, str):
                raise RuntimeError(f"Error in inference server:\n{response}")
            return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None
