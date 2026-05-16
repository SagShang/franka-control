"""Observation and action adapters for OpenPI Franka policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OpenPIObservationConfig:
    """Mapping from franka-control data to OpenPI pi0-style observation."""

    high_camera: str = "base_camera"
    wrist_camera: str = "wrist_camera"
    prompt: str = "manipulation"
    image_size: tuple[int, int] | None = None
    gripper_type: str = "robotiq"
    gripper_max_width: float = 0.08
    gripper_invert_state: bool = False


@dataclass(frozen=True)
class OpenPIActionConfig:
    """Mapping from an OpenPI action vector to a FrankaEnv action."""

    mode: str = "joint_abs"
    include_gripper: bool = True
    gripper_invert: bool = False


def make_openpi_observation(
    robot_obs: dict[str, Any],
    images: dict[str, np.ndarray],
    config: OpenPIObservationConfig,
) -> dict[str, Any]:
    """Build the OpenPI observation schema from robot state and RGB images.

    The returned schema matches the pi0/pi0.5 Franka joint-position convention:
    ``{"images": {"cam_high", "cam_wrist"}, "state": (8,), "prompt": str}``.
    """
    joint = _as_vector(robot_obs["joint_pos"], length=7, key="joint_pos")
    gripper = _openpi_gripper_state(robot_obs, config)
    state = np.ascontiguousarray(np.concatenate([joint, gripper]), dtype=np.float32)

    return {
        "images": {
            "cam_high": _as_rgb_uint8_image(
                _image_value(images, config.high_camera),
                config.image_size,
                "cam_high",
            ),
            "cam_wrist": _as_rgb_uint8_image(
                _image_value(images, config.wrist_camera),
                config.image_size,
                "cam_wrist",
            ),
        },
        "state": state,
        "prompt": config.prompt,
    }


def openpi_action_to_env_action(
    action: np.ndarray,
    env_action_low: np.ndarray,
    env_action_high: np.ndarray,
    config: OpenPIActionConfig,
) -> np.ndarray:
    """Convert one OpenPI policy action to a clipped ``FrankaEnv`` action."""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    robot_dim = _robot_action_dim(config.mode)
    expected_min = robot_dim + int(config.include_gripper)
    if action.shape[0] < expected_min:
        raise ValueError(
            f"OpenPI action for mode {config.mode!r} must have at least "
            f"{expected_min} values, got {action.shape}"
        )

    env_action = np.zeros(expected_min, dtype=np.float32)
    env_action[:robot_dim] = action[:robot_dim]

    if config.include_gripper:
        gripper_index = robot_dim
        gripper = float(np.clip(action[robot_dim], 0.0, 1.0))
        if config.gripper_invert:
            gripper = 1.0 - gripper
        low = float(env_action_low[gripper_index])
        high = float(env_action_high[gripper_index])
        gripper = low + gripper * (high - low)
        env_action[gripper_index] = gripper

    return np.clip(env_action, env_action_low, env_action_high).astype(np.float32)


def _openpi_gripper_state(
    robot_obs: dict[str, Any],
    config: OpenPIObservationConfig,
) -> np.ndarray:
    if "gripper_position" not in robot_obs:
        return np.asarray([0.0], dtype=np.float32)

    position = float(np.asarray(robot_obs["gripper_position"]).reshape(-1)[0])
    if config.gripper_type == "robotiq":
        value = np.clip(position / 255.0, 0.0, 1.0)
    else:
        max_width = max(float(config.gripper_max_width), np.finfo(np.float32).eps)
        value = 1.0 - np.clip(position / max_width, 0.0, 1.0)

    if config.gripper_invert_state:
        value = 1.0 - value
    return np.asarray([value], dtype=np.float32)


def _robot_action_dim(mode: str) -> int:
    if mode.startswith("joint"):
        return 7
    if mode.startswith("ee"):
        return 6
    raise ValueError(f"Unknown action mode {mode!r}")


def _image_value(images: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key in images:
        return images[key]
    raise KeyError(
        f"Camera image {key!r} not found; available cameras: {sorted(images)}"
    )


def _as_rgb_uint8_image(
    image: Any,
    image_size: tuple[int, int] | None,
    key: str,
) -> np.ndarray:
    image = _as_uint8(np.asarray(image))

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif (
        image.ndim == 3
        and image.shape[0] in (1, 3, 4)
        and image.shape[-1] not in (1, 3, 4)
    ):
        image = np.moveaxis(image, 0, -1)

    if image.ndim != 3:
        raise ValueError(f"{key} must be an image array, got shape {image.shape}")

    channels = image.shape[-1]
    if channels == 1:
        image = np.repeat(image, 3, axis=-1)
    elif channels == 4:
        image = image[..., :3]
    elif channels != 3:
        raise ValueError(f"{key} must have 1, 3, or 4 channels, got {image.shape}")

    if image_size is not None and image.shape[:2] != image_size:
        image = _resize_image(image, height=image_size[0], width=image_size[1])

    return np.ascontiguousarray(image, dtype=np.uint8)


def _as_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _resize_image(image: np.ndarray, *, height: int, width: int) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required when --image-size changes image shape"
        ) from exc
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _as_vector(value: Any, *, length: int, key: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (length,):
        raise ValueError(f"{key} must have shape ({length},), got {vector.shape}")
    return vector
