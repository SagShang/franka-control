"""Robotiq 2F native ZMQ server.

Runs on the control machine. This server exposes Robotiq-native operations
instead of pretending the gripper is a Franka Hand.

Native command semantics:
  - activate/reset map to Robotiq activation/reset
  - move/open/close use Robotiq position bits: 0=open, 255=closed
  - speed and force are Robotiq register values: 0..255
  - status returns gOBJ/gSTA/gGTO/gACT/kFLT/gFLT/gPR/gPO/gCU when available

Usage:
    python -m franka_control.robotiq --serial-port /dev/ttyUSB0
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import msgpack
import zmq

logger = logging.getLogger(__name__)


DEFAULT_CMD_PORT = 5556
DEFAULT_POLL_HZ = 200.0
DEFAULT_DEVICE_ID = 9
DEFAULT_SPEED = 255
DEFAULT_FORCE = 128
DEFAULT_TIMEOUT = 5.0
DEFAULT_MAX_WIDTH_M = 0.085


def _clamp_byte(value: int | float) -> int:
    return min(max(int(round(float(value))), 0), 255)


class RobotiqStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"


@dataclass
class CachedRobotiqState:
    """Thread-safe cached Robotiq state."""

    position: int = 0
    requested_position: int | None = None
    speed: int | None = None
    force: int | None = None
    width_m: float | None = None
    max_width_m: float = DEFAULT_MAX_WIDTH_M
    object_status: int | None = None
    activated: bool | None = None
    started: bool | None = None
    fault: int | None = None
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "requested_position": self.requested_position,
            "speed": self.speed,
            "force": self.force,
            "width_m": self.width_m,
            "max_width_m": self.max_width_m,
            "object_status": self.object_status,
            "activated": self.activated,
            "started": self.started,
            "fault": self.fault,
            "raw": self.raw,
        }


class RobotiqHardware:
    """Small wrapper around pyrobotiqgripper.RobotiqGripper."""

    def __init__(
        self,
        serial_port: str = "auto",
        device_id: int = DEFAULT_DEVICE_ID,
        connection_type: str = "RTU",
        tcp_host: str = "127.0.0.1",
        tcp_port: int = 54321,
        debug: bool = False,
        max_width_m: float = DEFAULT_MAX_WIDTH_M,
        gripper_cls: Any | None = None,
    ):
        self.serial_port = serial_port
        self.device_id = int(device_id)
        self.connection_type = connection_type
        self.tcp_host = tcp_host
        self.tcp_port = int(tcp_port)
        self.debug = debug
        self.max_width_m = float(max_width_m)
        self._gripper_cls = gripper_cls
        self._gripper: Any = None
        self._last_speed: int | None = None
        self._last_force: int | None = None
        self._last_requested_position: int | None = None

    def connect(self) -> None:
        if self._gripper_cls is None:
            try:
                from pyrobotiqgripper import RobotiqGripper
            except ImportError as exc:
                raise RuntimeError(
                    "pyrobotiqgripper is not installed. Install "
                    "franka-control[control-machine] or run "
                    "pip install pyrobotiqgripper"
                ) from exc
            self._gripper_cls = RobotiqGripper

        self._gripper = self._gripper_cls(
            com_port=self.serial_port,
            device_id=self.device_id,
            connection_type=self.connection_type,
            tcp_host=self.tcp_host,
            tcp_port=self.tcp_port,
            debug=self.debug,
        )
        if hasattr(self._gripper, "connect"):
            self._gripper.connect()

    def disconnect(self) -> None:
        if self._gripper is not None and hasattr(self._gripper, "disconnect"):
            self._gripper.disconnect()

    def reset(self) -> bool:
        self._require_connected()
        self._gripper.reset()
        self._last_requested_position = None
        return True

    def activate(
        self,
        reset: bool = True,
        start: bool = True,
        open_after: bool = False,
        speed: int = DEFAULT_SPEED,
        force: int = DEFAULT_FORCE,
    ) -> bool:
        self._require_connected()
        logger.warning(
            "Activating Robotiq gripper. Keep the fingers clear; activation may move."
        )
        self._gripper.activate(
            reset=bool(reset),
            start=bool(start),
            refreshStatus=True,
        )
        if open_after:
            self.move(0, speed=speed, force=force, wait=True)
        return True

    def move(
        self,
        position: int,
        speed: int = DEFAULT_SPEED,
        force: int = DEFAULT_FORCE,
        wait: bool = True,
    ) -> bool:
        self._require_connected()
        position = _clamp_byte(position)
        speed = _clamp_byte(speed)
        force = _clamp_byte(force)
        self._last_requested_position = position
        self._last_speed = speed
        self._last_force = force
        self._gripper.move(
            position,
            speed=speed,
            force=force,
            wait=wait,
            readStatus=True,
            refreshStatus=False,
        )
        return True

    def open(
        self,
        speed: int = DEFAULT_SPEED,
        force: int = DEFAULT_FORCE,
        wait: bool = True,
    ) -> bool:
        return self.move(0, speed=speed, force=force, wait=wait)

    def close(
        self,
        speed: int = DEFAULT_SPEED,
        force: int = DEFAULT_FORCE,
        wait: bool = True,
    ) -> bool:
        return self.move(255, speed=speed, force=force, wait=wait)

    def stop(self) -> bool:
        self._require_connected()
        self._gripper.stop()
        return True

    def read_state(self) -> CachedRobotiqState:
        self._require_connected()
        raw = {}
        try:
            raw = self._gripper.status(refreshStatus=True) or {}
        except Exception:
            logger.debug("Robotiq status() failed; falling back to position()",
                         exc_info=True)

        position = raw.get("gPO")
        if position is None:
            position = self._gripper.position(refreshStatus=True)
        position = _clamp_byte(position)

        return CachedRobotiqState(
            position=position,
            requested_position=_int_or_none(raw.get("gPR"))
            if raw else self._last_requested_position,
            speed=self._last_speed,
            force=self._last_force,
            width_m=self.position_to_width_m(position),
            max_width_m=self.max_width_m,
            object_status=_int_or_none(raw.get("gOBJ")),
            activated=_bool_or_none(raw.get("gACT")),
            started=_bool_or_none(raw.get("gGTO")),
            fault=_int_or_none(raw.get("gFLT")),
            raw=_clean_raw_status(raw),
        )

    def position_to_width_m(self, position: int) -> float:
        position = _clamp_byte(position)
        return self.max_width_m * (1.0 - position / 255.0)

    def _require_connected(self) -> None:
        if self._gripper is None:
            raise RuntimeError("Robotiq gripper is not connected")


def _int_or_none(value) -> int | None:
    if value is None:
        return None
    return int(value)


def _bool_or_none(value) -> bool | None:
    if value is None:
        return None
    return bool(int(value))


def _clean_raw_status(raw: dict) -> dict:
    clean = {}
    for key, value in raw.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and value.is_integer():
            clean[key] = int(value)
        else:
            clean[key] = value
    return clean


class RobotiqServer:
    """ZMQ server exposing Robotiq-native operations."""

    def __init__(
        self,
        serial_port: str = "auto",
        cmd_port: int = DEFAULT_CMD_PORT,
        poll_hz: float = DEFAULT_POLL_HZ,
        device_id: int = DEFAULT_DEVICE_ID,
        connection_type: str = "RTU",
        tcp_host: str = "127.0.0.1",
        tcp_port: int = 54321,
        max_width_m: float = DEFAULT_MAX_WIDTH_M,
        default_speed: int = DEFAULT_SPEED,
        default_force: int = DEFAULT_FORCE,
        debug: bool = False,
        hardware: RobotiqHardware | None = None,
    ):
        self.serial_port = serial_port
        self.cmd_port = int(cmd_port)
        self.poll_hz = max(1.0, min(float(poll_hz), 200.0))
        self._poll_interval = 1.0 / self.poll_hz
        self.default_speed = _clamp_byte(default_speed)
        self.default_force = _clamp_byte(default_force)
        self._hardware = hardware or RobotiqHardware(
            serial_port=serial_port,
            device_id=device_id,
            connection_type=connection_type,
            tcp_host=tcp_host,
            tcp_port=tcp_port,
            debug=debug,
            max_width_m=max_width_m,
        )
        self._hardware_lock = threading.Lock()

        self._cached_state = CachedRobotiqState(max_width_m=max_width_m)
        self._state_lock = threading.Lock()

        self._worker_lock = threading.Lock()
        self._worker_status = RobotiqStatus.IDLE
        self._worker_result: dict | None = None
        self._worker_thread: threading.Thread | None = None
        self._stop_flag = False

        self._ctx: zmq.Context | None = None
        self._cmd_socket = None
        self._running = False
        self._poll_thread: threading.Thread | None = None

    def start(self) -> None:
        logger.info("Connecting to Robotiq gripper on %s ...", self.serial_port)
        self._hardware.connect()
        logger.info("Robotiq gripper connected.")

        self._ctx = zmq.Context()
        self._cmd_socket = self._ctx.socket(zmq.ROUTER)
        self._cmd_socket.bind(f"tcp://*:{self.cmd_port}")
        self._cmd_socket.setsockopt(zmq.RCVTIMEO, 200)
        logger.info("Robotiq server listening on port %d", self.cmd_port)

        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True
        )
        self._poll_thread.start()

    def run(self) -> None:
        while self._running:
            try:
                parts = self._cmd_socket.recv_multipart()
            except zmq.Again:
                continue

            if len(parts) < 3:
                continue

            identity = parts[0]
            data = parts[2]
            response = self._dispatch(data)
            packed = msgpack.packb(response, use_bin_type=True)
            self._cmd_socket.send_multipart([identity, b"", packed])

        self._cleanup()

    def _cleanup(self) -> None:
        self._running = False
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)

        if self._cmd_socket is not None:
            try:
                self._cmd_socket.close()
            except Exception:
                pass
            self._cmd_socket = None

        if self._ctx is not None:
            try:
                self._ctx.term()
            except Exception:
                pass
            self._ctx = None

        try:
            self._hardware.disconnect()
        except Exception:
            logger.debug("Robotiq disconnect failed", exc_info=True)
        logger.info("Robotiq server stopped.")

    def _dispatch(self, raw: bytes) -> dict:
        try:
            msg = msgpack.unpackb(raw, raw=False)
        except Exception:
            return {"success": False, "error": "Invalid msgpack"}

        command = msg.get("command", "")
        params = msg.get("params", {})
        handlers = {
            "activate": self._cmd_activate,
            "reset": self._cmd_reset,
            "move": self._cmd_move,
            "open": self._cmd_open,
            "close": self._cmd_close,
            "stop": self._cmd_stop,
            "get_state": self._cmd_get_state,
            "shutdown": self._cmd_shutdown,
        }
        handler = handlers.get(command)
        if handler is None:
            return {"success": False, "error": f"Unknown command: {command}"}

        try:
            return handler(params)
        except Exception as e:
            logger.exception("Error handling '%s'", command)
            return {"success": False, "error": str(e)}

    def _cmd_get_state(self, params: dict) -> dict:
        with self._state_lock:
            state = self._cached_state.to_dict()
        with self._worker_lock:
            state["status"] = self._worker_status.value
            result = self._worker_result
        return {"success": True, "state": state, "result": result}

    def _cmd_activate(self, params: dict) -> dict:
        return self._submit_job(
            "activate",
            lambda: self._call_hardware(
                lambda: self._hardware.activate(
                    reset=bool(params.get("reset", True)),
                    start=bool(params.get("start", True)),
                    open_after=bool(params.get("open_after", False)),
                    speed=self._speed_param(params),
                    force=self._force_param(params),
                )
            ),
        )

    def _cmd_reset(self, params: dict) -> dict:
        return self._submit_job(
            "reset",
            lambda: self._call_hardware(lambda: self._hardware.reset()),
        )

    def _cmd_move(self, params: dict) -> dict:
        if "position" not in params:
            return {"success": False, "error": "move requires position"}
        position = _clamp_byte(params["position"])
        wait = bool(params.get("wait", True))
        return self._submit_job(
            "move",
            lambda: self._call_hardware(
                lambda: self._hardware.move(
                    position=position,
                    speed=self._speed_param(params),
                    force=self._force_param(params),
                    wait=wait,
                )
            ),
        )

    def _cmd_open(self, params: dict) -> dict:
        wait = bool(params.get("wait", True))
        return self._submit_job(
            "open",
            lambda: self._call_hardware(
                lambda: self._hardware.open(
                    speed=self._speed_param(params),
                    force=self._force_param(params),
                    wait=wait,
                )
            ),
        )

    def _cmd_close(self, params: dict) -> dict:
        wait = bool(params.get("wait", True))
        return self._submit_job(
            "close",
            lambda: self._call_hardware(
                lambda: self._hardware.close(
                    speed=self._speed_param(params),
                    force=self._force_param(params),
                    wait=wait,
                )
            ),
        )

    def _cmd_stop(self, params: dict) -> dict:
        self._stop_flag = True
        try:
            self._hardware.stop()
        except Exception:
            pass
        with self._worker_lock:
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=2.0)
            self._worker_status = RobotiqStatus.IDLE
            self._worker_result = None
            self._worker_thread = None
            self._stop_flag = False
        self._refresh_cached_state_after_job()
        return {"success": True}

    def _cmd_shutdown(self, params: dict) -> dict:
        self._running = False
        return {"success": True}

    def _submit_job(self, name: str, fn) -> dict:
        with self._worker_lock:
            if self._worker_status == RobotiqStatus.BUSY:
                return {"success": False, "error": "Robotiq busy"}
            self._worker_status = RobotiqStatus.BUSY
            self._worker_result = None

        def _run():
            try:
                result = fn()
                if not self._stop_flag:
                    self._refresh_cached_state_after_job()
                with self._worker_lock:
                    if not self._stop_flag:
                        self._worker_result = {
                            "success": True if result is None else bool(result),
                        }
                    self._worker_status = RobotiqStatus.IDLE
            except Exception as e:
                with self._worker_lock:
                    if not self._stop_flag:
                        self._worker_result = {"success": False, "error": str(e)}
                    self._worker_status = RobotiqStatus.IDLE

        t = threading.Thread(target=_run, daemon=True)
        self._worker_thread = t
        t.start()
        return {"success": True, "accepted": True}

    def _poll_once(self) -> None:
        with self._worker_lock:
            if self._worker_status == RobotiqStatus.BUSY:
                return
        try:
            state = self._read_hardware_state()
            self._update_cached_state(state)
        except Exception as e:
            logger.warning("Robotiq state poll failed: %s", e)

    def _poll_loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            self._poll_once()
            elapsed = time.monotonic() - t0
            remaining = self._poll_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _call_hardware(self, fn):
        with self._hardware_lock:
            return fn()

    def _read_hardware_state(self) -> CachedRobotiqState:
        with self._hardware_lock:
            return self._hardware.read_state()

    def _update_cached_state(self, state: CachedRobotiqState) -> None:
        with self._state_lock:
            self._cached_state = state

    def _refresh_cached_state_after_job(self) -> None:
        try:
            self._update_cached_state(self._read_hardware_state())
        except Exception:
            logger.debug("Robotiq state refresh after job failed", exc_info=True)

    def _speed_param(self, params: dict) -> int:
        return _clamp_byte(params.get("speed", self.default_speed))

    def _force_param(self, params: dict) -> int:
        return _clamp_byte(params.get("force", self.default_force))


def main():
    parser = argparse.ArgumentParser(description="Robotiq 2F native ZMQ Server")
    parser.add_argument("--serial-port", default="auto",
                        help="Serial port, e.g. /dev/ttyUSB0 (default: auto)")
    parser.add_argument("--device-id", type=int, default=DEFAULT_DEVICE_ID,
                        help=f"Modbus device id (default: {DEFAULT_DEVICE_ID})")
    parser.add_argument("--connection-type", default="RTU",
                        choices=["RTU", "RTU_VIA_TCP"])
    parser.add_argument("--tcp-host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=54321)
    parser.add_argument("--max-width-m", type=float, default=DEFAULT_MAX_WIDTH_M,
                        help="Physical max opening for observation only")
    parser.add_argument("--default-speed", type=int, default=DEFAULT_SPEED,
                        help=f"Robotiq speed 0..255 (default: {DEFAULT_SPEED})")
    parser.add_argument("--default-force", type=int, default=DEFAULT_FORCE,
                        help=f"Robotiq force 0..255 (default: {DEFAULT_FORCE})")
    parser.add_argument("--port", type=int, default=DEFAULT_CMD_PORT,
                        help=f"ZMQ port (default: {DEFAULT_CMD_PORT})")
    parser.add_argument("--poll-hz", type=float, default=DEFAULT_POLL_HZ,
                        help=f"State poll Hz (default: {DEFAULT_POLL_HZ})")
    parser.add_argument("--debug-modbus", action="store_true",
                        help="Enable pyrobotiqgripper/pymodbus debug logs")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    server = RobotiqServer(
        serial_port=args.serial_port,
        cmd_port=args.port,
        poll_hz=args.poll_hz,
        device_id=args.device_id,
        connection_type=args.connection_type,
        tcp_host=args.tcp_host,
        tcp_port=args.tcp_port,
        max_width_m=args.max_width_m,
        default_speed=args.default_speed,
        default_force=args.default_force,
        debug=args.debug_modbus,
    )

    def _signal_handler(sig, frame):
        logger.info("Received signal %d, shutting down ...", sig)
        server._running = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    server.start()
    server.run()


if __name__ == "__main__":
    main()
