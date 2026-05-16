"""Offline tests for FrankaEnv action-space and clipping logic."""

import numpy as np
import pytest

from franka_control.envs.franka_env import (
    FrankaEnv,
    JOINT_LIMIT_HIGH,
    JOINT_LIMIT_LOW,
)


def test_invalid_action_mode_raises_before_hardware_connection():
    with pytest.raises(ValueError, match="action_mode"):
        FrankaEnv(robot_ip="127.0.0.1", action_mode="invalid")


def test_action_space_dimensions_with_and_without_gripper():
    env_no_gripper = FrankaEnv(
        robot_ip="127.0.0.1",
        gripper_host=None,
        action_mode="ee_delta",
    )
    assert env_no_gripper.action_space.shape == (6,)

    env_with_gripper = FrankaEnv(
        robot_ip="127.0.0.1",
        gripper_host="127.0.0.1",
        action_mode="joint_abs",
        gripper_mode="binary",
    )
    assert env_with_gripper.action_space.shape == (8,)
    assert env_with_gripper.action_space.low[-1] == 0.0
    assert env_with_gripper.action_space.high[-1] == 1.0

    env_with_robotiq = FrankaEnv(
        robot_ip="127.0.0.1",
        gripper_host="127.0.0.1",
        gripper_type="robotiq",
        action_mode="ee_delta",
        gripper_mode="continuous",
    )
    assert env_with_robotiq.action_space.shape == (7,)
    assert env_with_robotiq.action_space.low[-1] == 0.0
    assert env_with_robotiq.action_space.high[-1] == 255.0
    assert "gripper_position" in env_with_robotiq.observation_space.spaces
    assert "robotiq_position" not in env_with_robotiq.observation_space.spaces


def test_invalid_gripper_type_raises_before_hardware_connection():
    with pytest.raises(ValueError, match="gripper_type"):
        FrankaEnv(
            robot_ip="127.0.0.1",
            gripper_host="127.0.0.1",
            gripper_type="not-a-gripper",
        )


def test_joint_targets_clip_to_fr3_joint_limits():
    env = FrankaEnv(robot_ip="127.0.0.1", action_mode="joint_abs")

    high = env._clip_robot_target(np.full(7, 999.0))
    low = env._clip_robot_target(np.full(7, -999.0))

    assert np.allclose(high, JOINT_LIMIT_HIGH)
    assert np.allclose(low, JOINT_LIMIT_LOW)


def test_robotiq_binary_uses_native_close_not_franka_grasp():
    class FakeRobotiqClient:
        def __init__(self):
            self.calls = []

        def open(self, speed=255, force=128):
            self.calls.append(("open", speed, force))
            return True

        def close(self, speed=255, force=128):
            self.calls.append(("close", speed, force))
            return True

        def move(self, position, speed=255, force=128, wait=False):
            self.calls.append(("move", position, speed, force, wait))
            return True

        def grasp(self, *args, **kwargs):
            raise AssertionError("Robotiq path must not call Franka grasp")

    env = FrankaEnv(
        robot_ip="127.0.0.1",
        gripper_host="127.0.0.1",
        gripper_type="robotiq",
        gripper_mode="binary",
    )
    env._gripper = FakeRobotiqClient()

    env._apply_gripper_action(0.0)
    env._apply_gripper_action(1.0)

    assert env._gripper.calls == [
        ("close", 255, 128),
        ("open", 255, 128),
    ]
    assert env._cached_robotiq_position == 0


def test_robotiq_continuous_rejection_is_nonfatal():
    class BusyRobotiqClient:
        def __init__(self):
            self.calls = []

        def move(self, position, speed=255, force=128, wait=False):
            self.calls.append((position, speed, force, wait))
            return False

    env = FrankaEnv(
        robot_ip="127.0.0.1",
        gripper_host="127.0.0.1",
        gripper_type="robotiq",
        gripper_mode="continuous",
    )
    env._gripper = BusyRobotiqClient()

    env._apply_gripper_action(42)

    assert env._gripper.calls == [(42, 255, 128, False)]
