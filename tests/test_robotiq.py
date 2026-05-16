"""Tests for Robotiq-native server/client semantics."""

import os
import sys
import threading
import time

import msgpack
import pytest
import zmq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from franka_control.robotiq.robotiq_client import RobotiqClient
from franka_control.robotiq.robotiq_server import (
    CachedRobotiqState,
    RobotiqServer,
    RobotiqStatus,
)


class FakeRobotiqHardware:
    def __init__(self):
        self.connected = False
        self.disconnected = False
        self.activated = []
        self.reset_count = 0
        self.moves = []
        self.position = 0
        self.stopped = False
        self.max_width_m = 0.085

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.disconnected = True

    def activate(self, reset=True, start=True, open_after=False, speed=255, force=128):
        self.activated.append({
            "reset": reset,
            "start": start,
            "open_after": open_after,
            "speed": speed,
            "force": force,
        })
        if open_after:
            self.move(0, speed=speed, force=force, wait=True)
        return True

    def reset(self):
        self.reset_count += 1
        return True

    def move(self, position, speed=255, force=128, wait=True):
        self.position = int(position)
        self.moves.append({
            "position": int(position),
            "speed": int(speed),
            "force": int(force),
            "wait": bool(wait),
        })
        return True

    def open(self, speed=255, force=128, wait=True):
        return self.move(0, speed=speed, force=force, wait=wait)

    def close(self, speed=255, force=128, wait=True):
        return self.move(255, speed=speed, force=force, wait=wait)

    def stop(self):
        self.stopped = True
        return True

    def read_state(self):
        return CachedRobotiqState(
            position=self.position,
            requested_position=self.position,
            speed=self.moves[-1]["speed"] if self.moves else None,
            force=self.moves[-1]["force"] if self.moves else None,
            width_m=self.max_width_m * (1.0 - self.position / 255.0),
            max_width_m=self.max_width_m,
            object_status=0,
            activated=True,
            started=True,
            fault=0,
            raw={
                "gPO": self.position,
                "gPR": self.position,
                "gOBJ": 0,
                "gACT": 1,
                "gGTO": 1,
                "gFLT": 0,
            },
        )


def _pack(command, params=None):
    msg = {"command": command}
    if params:
        msg["params"] = params
    return msgpack.packb(msg, use_bin_type=True)


def _find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class TestRobotiqServerDispatch:
    def setup_method(self):
        self.hardware = FakeRobotiqHardware()
        self.server = RobotiqServer(hardware=self.hardware, cmd_port=0)

    def test_unknown_command(self):
        resp = self.server._dispatch(_pack("homing"))
        assert resp["success"] is False
        assert "Unknown command" in resp["error"]

    def test_move_requires_native_position(self):
        resp = self.server._dispatch(_pack("move", {"width": 0.02}))
        assert resp["success"] is False
        assert "position" in resp["error"]

    def test_move_clamps_position_speed_force(self):
        resp = self.server._dispatch(_pack(
            "move", {"position": 300, "speed": 999, "force": -5, "wait": False}
        ))
        assert resp["success"] is True
        time.sleep(0.2)
        assert self.hardware.moves[-1] == {
            "position": 255,
            "speed": 255,
            "force": 0,
            "wait": False,
        }

    def test_activate_is_native_not_homing(self):
        resp = self.server._dispatch(_pack(
            "activate", {"reset": False, "start": True, "open_after": True}
        ))
        assert resp["success"] is True
        time.sleep(0.2)
        assert self.hardware.activated[-1]["reset"] is False
        assert self.hardware.moves[-1]["position"] == 0

    def test_state_shape(self):
        self.hardware.move(128, speed=10, force=20)
        self.server._update_cached_state(self.hardware.read_state())
        resp = self.server._dispatch(_pack("get_state"))
        assert resp["success"] is True
        state = resp["state"]
        assert state["position"] == 128
        assert state["width_m"] == pytest.approx(0.085 * (1.0 - 128 / 255.0))
        assert state["raw"]["gPO"] == 128
        assert state["status"] == RobotiqStatus.IDLE.value


class TestRobotiqClientEndToEnd:
    def setup_method(self):
        self.port = _find_free_port()
        self.hardware = FakeRobotiqHardware()
        self.server = RobotiqServer(hardware=self.hardware, cmd_port=self.port)
        self.server._ctx = zmq.Context()
        self.server._cmd_socket = self.server._ctx.socket(zmq.ROUTER)
        self.server._cmd_socket.bind(f"tcp://*:{self.port}")
        self.server._cmd_socket.setsockopt(zmq.RCVTIMEO, 200)
        self.server._running = True
        self._server_thread = threading.Thread(target=self.server.run, daemon=True)
        self._server_thread.start()
        time.sleep(0.1)
        self.client = RobotiqClient("127.0.0.1", port=self.port)

    def teardown_method(self):
        self.client.disconnect()
        self.server._running = False
        time.sleep(0.3)
        self.server._cleanup()

    def test_open_close_move_and_state(self):
        assert self.client.activate(reset=False, open_after=False, timeout=2.0) is True
        assert self.client.move(42, speed=77, force=88, timeout=2.0) is True
        assert self.hardware.moves[-1]["position"] == 42
        assert self.client.open(timeout=2.0) is True
        assert self.hardware.moves[-1]["position"] == 0
        assert self.client.close(timeout=2.0) is True
        assert self.hardware.moves[-1]["position"] == 255
        state = self.client.get_state()
        assert state["position"] == 255
        assert state["width_m"] == pytest.approx(0.0)

    def test_shutdown(self):
        assert self.client.shutdown_server() is True
