"""Offline tests for GELLO teleop adapter."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from franka_control.teleop.gello_teleop import GelloTeleop


class _FakeDynamixelDriver:
    joints = np.zeros(8, dtype=np.float64)
    init_args = None

    def __init__(self, ids, port="/dev/ttyUSB0", baudrate=57600, **kwargs):
        self.ids = ids
        self.port = port
        self.baudrate = baudrate
        self.kwargs = kwargs
        self.closed = False
        self.torque_modes = []
        _FakeDynamixelDriver.init_args = (ids, port, baudrate, kwargs)

    def get_joints(self):
        return _FakeDynamixelDriver.joints.copy()

    def set_torque_mode(self, enable):
        self.torque_modes.append(enable)

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _fake_gello_modules(monkeypatch):
    driver_module = types.ModuleType("gello.dynamixel.driver")
    driver_module.DynamixelDriver = _FakeDynamixelDriver

    dynamixel_module = types.ModuleType("gello.dynamixel")
    gello_module = types.ModuleType("gello")

    monkeypatch.setitem(sys.modules, "gello", gello_module)
    monkeypatch.setitem(sys.modules, "gello.dynamixel", dynamixel_module)
    monkeypatch.setitem(sys.modules, "gello.dynamixel.driver", driver_module)
    _FakeDynamixelDriver.joints = np.zeros(8, dtype=np.float64)
    _FakeDynamixelDriver.init_args = None


def _write_config(tmp_path):
    path = tmp_path / "franka_gello_single.yaml"
    path.write_text(
        """
SINGLE:
  com_port: "usb-FAKE"
  num_arm_joints: 7
  joint_signs: [1, -1, 1, -1, 1, 1, 1]
  gripper: true
  assembly_offsets: [0, 0, 0, 0, 0, 0, 0]
  gripper_range_rad: [0.2, 1.2]
""",
        encoding="utf-8",
    )
    return path


def test_gello_teleop_loads_config_and_returns_joint_abs_action(tmp_path):
    config = _write_config(tmp_path)
    _FakeDynamixelDriver.joints = np.array(
        [0.1, 0.2, -0.3, -1.0, 0.5, 1.2, -0.7, 0.9],
        dtype=np.float64,
    )

    teleop = GelloTeleop(config_path=config)
    try:
        action, info = teleop.get_action()
    finally:
        teleop.close()

    assert action.shape == (8,)
    assert action[-1] == pytest.approx(0.056)
    assert info["intervened"] is True
    assert info["gripper"] == pytest.approx(0.056)
    assert info["gripper_percent"] == pytest.approx(0.7)

    ids, port, baudrate, kwargs = _FakeDynamixelDriver.init_args
    assert ids == list(range(1, 9))
    assert port == "/dev/serial/by-id/usb-FAKE"
    assert baudrate == 57600
    assert kwargs["use_fake_fallback"] is False


def test_gello_teleop_supports_port_override_and_no_gripper(tmp_path):
    config = _write_config(tmp_path)
    teleop = GelloTeleop(
        config_path=config,
        port="/dev/ttyUSB42",
        use_gripper=False,
    )
    try:
        action, info = teleop.get_action()
    finally:
        teleop.close()

    assert action.shape == (7,)
    assert info["gripper"] is None
    assert _FakeDynamixelDriver.init_args[1] == "/dev/ttyUSB42"


def test_gello_teleop_can_output_robotiq_position_units(tmp_path):
    config = _write_config(tmp_path)
    _FakeDynamixelDriver.joints = np.array(
        [0.0, 0.0, 0.0, -1.0, 0.0, 1.2, 0.0, 0.9],
        dtype=np.float64,
    )

    teleop = GelloTeleop(
        config_path=config,
        gripper_output="robotiq_position",
    )
    try:
        action, info = teleop.get_action()
    finally:
        teleop.close()

    assert action[-1] == pytest.approx(76.5)
    assert info["gripper"] == pytest.approx(76.5)
    assert info["gripper_percent"] == pytest.approx(0.7)
    assert info["gripper_output"] == "robotiq_position"


def test_gello_teleop_requires_existing_config_section(tmp_path):
    config = _write_config(tmp_path)
    with pytest.raises(ValueError, match="section 'RIGHT'"):
        GelloTeleop(config_path=config, config_section="RIGHT")
