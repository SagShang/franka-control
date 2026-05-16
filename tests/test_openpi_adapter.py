from __future__ import annotations

import numpy as np

from franka_control.openpi.action_queue import ActionChunkQueue, actions_from_result
from franka_control.openpi.adapter import (
    OpenPIActionConfig,
    OpenPIObservationConfig,
    make_openpi_observation,
    openpi_action_to_env_action,
)


class FakePolicy:
    def __init__(self, horizon: int = 2) -> None:
        self.calls = []
        self.horizon = horizon

    def infer(self, observation):
        call_index = len(self.calls)
        self.calls.append(dict(observation))
        base = call_index * 100
        return {
            "actions": np.arange(
                base,
                base + self.horizon,
                dtype=np.float32,
            )[:, None]
        }


def test_action_chunk_queue_fetches_next_chunk_when_empty() -> None:
    policy = FakePolicy(horizon=2)
    queue = ActionChunkQueue(return_horizon=2)

    assert queue.next_action(policy, {"step": 0})[0] == 0
    assert queue.next_action(policy, {"step": 1})[0] == 1
    assert queue.next_action(policy, {"step": 2})[0] == 100

    assert len(policy.calls) == 2
    assert policy.calls[0]["_reset"] is True
    assert "_reset" not in policy.calls[1]


def test_actions_from_result_supports_prefixed_key() -> None:
    actions = actions_from_result(
        {"action.actions": np.array([[1.0, 2.0]], dtype=np.float32)},
        "actions",
    )

    np.testing.assert_array_equal(actions, [[1.0, 2.0]])


def test_make_openpi_observation_uses_camera_mapping_and_gripper_state() -> None:
    robot_obs = {
        "joint_pos": np.arange(7, dtype=np.float32),
        "gripper_position": np.array([255], dtype=np.float32),
    }
    images = {
        "base": np.ones((4, 5, 3), dtype=np.uint8),
        "wrist": np.zeros((4, 5, 3), dtype=np.uint8),
    }
    config = OpenPIObservationConfig(
        high_camera="base",
        wrist_camera="wrist",
        prompt="pick",
        gripper_type="robotiq",
    )

    obs = make_openpi_observation(robot_obs, images, config)

    assert obs["prompt"] == "pick"
    assert obs["images"]["cam_high"].shape == (4, 5, 3)
    assert obs["images"]["cam_wrist"].shape == (4, 5, 3)
    np.testing.assert_array_equal(obs["state"][:7], np.arange(7, dtype=np.float32))
    assert obs["state"][7] == 1.0


def test_openpi_binary_gripper_action_is_mapped_to_env_semantics() -> None:
    low = np.array([-10.0] * 7 + [0.0], dtype=np.float32)
    high = np.array([10.0] * 7 + [1.0], dtype=np.float32)
    action = np.array([0, 1, 2, 3, 4, 5, 6, 1.0], dtype=np.float32)

    env_action = openpi_action_to_env_action(
        action,
        low,
        high,
        OpenPIActionConfig(mode="joint_abs"),
    )

    np.testing.assert_array_equal(env_action[:7], action[:7])
    assert env_action[7] == 0.0


def test_openpi_action_can_skip_gripper() -> None:
    low = np.array([-0.1] * 7, dtype=np.float32)
    high = np.array([0.1] * 7, dtype=np.float32)
    action = np.array([0.2] * 7, dtype=np.float32)

    env_action = openpi_action_to_env_action(
        action,
        low,
        high,
        OpenPIActionConfig(mode="joint_delta", include_gripper=False),
    )

    np.testing.assert_allclose(env_action, high)


def test_openpi_continuous_gripper_scales_to_env_bounds() -> None:
    low = np.array([-10.0] * 7 + [0.0], dtype=np.float32)
    high = np.array([10.0] * 7 + [255.0], dtype=np.float32)
    action = np.array([0, 1, 2, 3, 4, 5, 6, 0.25], dtype=np.float32)

    env_action = openpi_action_to_env_action(
        action,
        low,
        high,
        OpenPIActionConfig(
            mode="joint_abs",
            binary_gripper=False,
            gripper_invert=False,
        ),
    )

    assert env_action[7] == 63.75
