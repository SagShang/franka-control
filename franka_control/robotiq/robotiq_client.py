"""Robotiq 2F native ZMQ client."""

from __future__ import annotations

import logging
import time
from typing import Optional

import msgpack
import zmq

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0


class RobotiqClient:
    """Client for the Robotiq native ZMQ server.

    Public methods use Robotiq-native units:
    - position: 0=open, 255=closed
    - speed: 0..255
    - force: 0..255
    """

    def __init__(
        self,
        host: str,
        port: int = 5556,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.RCVTIMEO, 10000)
        self._socket.setsockopt(zmq.SNDTIMEO, 5000)
        self._socket.connect(f"tcp://{host}:{port}")
        logger.info("Connected to Robotiq server at %s:%d", host, port)

    def _send_command(self, command: str, params: dict | None = None) -> dict:
        msg = {"command": command}
        if params:
            msg["params"] = params
        try:
            packed = msgpack.packb(msg, use_bin_type=True)
            self._socket.send_multipart([b"", packed])
            parts = self._socket.recv_multipart()
            if len(parts) < 2:
                return {"success": False, "error": "Malformed response"}
            return msgpack.unpackb(parts[-1], raw=False)
        except zmq.Again:
            logger.error("Timeout waiting for Robotiq server response")
            return {"success": False, "error": "Server timeout"}
        except Exception as e:
            logger.exception("Robotiq command failed")
            return {"success": False, "error": str(e)}

    def _wait_until_idle(self, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            resp = self._send_command("get_state")
            if not resp.get("success"):
                return resp
            state = resp.get("state", {})
            if state.get("status") == "idle":
                result = resp.get("result")
                if result is not None:
                    return result
                return {"success": True, "state": state}
            time.sleep(0.02)
        return {"success": False, "error": "Timeout waiting for Robotiq"}

    def activate(
        self,
        reset: bool = True,
        start: bool = True,
        open_after: bool = False,
        speed: int = 255,
        force: int = 128,
        timeout: float | None = None,
    ) -> bool:
        timeout = self.timeout if timeout is None else timeout
        resp = self._send_command("activate", {
            "reset": reset,
            "start": start,
            "open_after": open_after,
            "speed": speed,
            "force": force,
        })
        if not resp.get("success"):
            logger.error("Robotiq activate rejected: %s", resp.get("error"))
            return False
        result = self._wait_until_idle(timeout)
        return result.get("success", False)

    def reset(self, timeout: float | None = None) -> bool:
        timeout = self.timeout if timeout is None else timeout
        resp = self._send_command("reset")
        if not resp.get("success"):
            logger.error("Robotiq reset rejected: %s", resp.get("error"))
            return False
        result = self._wait_until_idle(timeout)
        return result.get("success", False)

    def move(
        self,
        position: int,
        speed: int = 255,
        force: int = 128,
        wait: bool = True,
        timeout: float | None = None,
    ) -> bool:
        resp = self._send_command("move", {
            "position": position,
            "speed": speed,
            "force": force,
            "wait": wait,
        })
        if not resp.get("success"):
            logger.error("Robotiq move rejected: %s", resp.get("error"))
            return False
        if not wait:
            return True
        timeout = self.timeout if timeout is None else timeout
        result = self._wait_until_idle(timeout)
        return result.get("success", False)

    def open(
        self,
        speed: int = 255,
        force: int = 128,
        wait: bool = True,
        timeout: float | None = None,
    ) -> bool:
        return self._simple_motion("open", speed, force, wait, timeout)

    def close(
        self,
        speed: int = 255,
        force: int = 128,
        wait: bool = True,
        timeout: float | None = None,
    ) -> bool:
        return self._simple_motion("close", speed, force, wait, timeout)

    def _simple_motion(
        self,
        command: str,
        speed: int,
        force: int,
        wait: bool,
        timeout: float | None,
    ) -> bool:
        resp = self._send_command(command, {
            "speed": speed,
            "force": force,
            "wait": wait,
        })
        if not resp.get("success"):
            logger.error("Robotiq %s rejected: %s", command, resp.get("error"))
            return False
        if not wait:
            return True
        timeout = self.timeout if timeout is None else timeout
        result = self._wait_until_idle(timeout)
        return result.get("success", False)

    def stop(self) -> bool:
        resp = self._send_command("stop")
        return resp.get("success", False)

    def get_state(self) -> Optional[dict]:
        resp = self._send_command("get_state")
        if resp.get("success"):
            return resp.get("state")
        return None

    def shutdown_server(self) -> bool:
        resp = self._send_command("shutdown")
        return resp.get("success", False)

    def disconnect(self) -> None:
        if self._socket:
            self._socket.close()
        if self._ctx:
            self._ctx.term()
        logger.info("Robotiq client disconnected.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
