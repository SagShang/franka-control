"""GELLO teleoperation provider for FrankaEnv.

Reads a GELLO leader arm through the upstream ``gello`` package and returns
absolute FR3 joint targets for ``FrankaEnv(action_mode="joint_abs")``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# From GELLO's FR3 ROS 2 implementation. These centers keep the leader joints
# in the same continuous branch as the physical FR3 joint range.
FR3_JOINT_LIMITS = np.array(
    [
        [-2.9007, 2.9007],
        [-1.8361, 1.8361],
        [-2.9007, 2.9007],
        [-3.0770, -0.1169],
        [-2.8763, 2.8763],
        [0.4398, 4.6216],
        [-3.0508, 3.0508],
    ],
    dtype=np.float64,
)
FR3_MID_JOINT_POSITIONS = FR3_JOINT_LIMITS.mean(axis=1)


def _maybe_add_gello_root(gello_root: str | Path | None) -> None:
    if gello_root is None:
        return
    root = Path(gello_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"GELLO root does not exist: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _load_config(config_path: str | Path, section: str) -> dict[str, Any]:
    path = Path(config_path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if section not in data:
        available = ", ".join(str(k) for k in data.keys()) or "<none>"
        raise ValueError(
            f"GELLO config section '{section}' not found in {path}. "
            f"Available sections: {available}"
        )
    config = dict(data[section] or {})
    if "com_port" not in config:
        raise ValueError(f"GELLO config section '{section}' is missing com_port")
    return config


def _resolve_port(com_port: str) -> str:
    if com_port.startswith("/"):
        return com_port
    return f"/dev/serial/by-id/{com_port}"


def _normalize_joint_positions(
    raw_positions: np.ndarray,
    assembly_offsets: np.ndarray,
    joint_signs: np.ndarray,
) -> np.ndarray:
    return (
        np.mod(
            (raw_positions - assembly_offsets) * joint_signs
            - FR3_MID_JOINT_POSITIONS,
            2 * np.pi,
        )
        - np.pi
        + FR3_MID_JOINT_POSITIONS
    )


class GelloTeleop:
    """GELLO leader-arm teleoperation provider.

    Args:
        config_path: YAML config path. This can be one of GELLO's ROS 2 config
            files, for example ``franka_gello_single.yaml``.
        config_section: Top-level YAML section to read, usually ``SINGLE``,
            ``LEFT``, or ``RIGHT``.
        port: Optional serial device override. If omitted, uses ``com_port``
            from the config.
        gello_root: Optional path to the upstream ``gello_software`` checkout.
            Useful when the package has not been installed into the environment.
        gripper_mode: ``"continuous"``, ``"binary"``, or None. Continuous mode
            maps GELLO open-width percentage to the selected output units.
        gripper_output: ``"franka_hand_width"`` outputs meters where 0.08 is
            open. ``"robotiq_position"`` outputs native Robotiq position bits
            where 0 is open and 255 is closed.
        gripper_max_width: Maximum Franka Hand width in meters for continuous
            mode.
        binary_gripper_threshold: Threshold used only in binary mode.
    """

    def __init__(
        self,
        config_path: str | Path,
        config_section: str = "SINGLE",
        port: str | None = None,
        gello_root: str | Path | None = None,
        gripper_mode: str | None = "continuous",
        gripper_output: str = "franka_hand_width",
        gripper_max_width: float = 0.08,
        binary_gripper_threshold: float = 0.5,
    ):
        _maybe_add_gello_root(gello_root)
        try:
            from gello.dynamixel.driver import DynamixelDriver
        except ImportError as exc:
            raise ImportError(
                "The upstream 'gello' package is required for GELLO teleop. "
                "Install it with 'python -m pip install -e /path/to/gello_software' "
                "or pass --gello-root /path/to/gello_software."
            ) from exc

        config = _load_config(config_path, config_section)
        self._num_arm_joints = int(config.get("num_arm_joints", 7))
        if self._num_arm_joints != 7:
            raise ValueError(
                "GelloTeleop currently supports a 7-DoF Franka GELLO arm; "
                f"got num_arm_joints={self._num_arm_joints}"
            )

        self._joint_signs = np.asarray(config["joint_signs"], dtype=np.float64)
        self._assembly_offsets = np.asarray(
            config["assembly_offsets"], dtype=np.float64
        )
        if self._joint_signs.shape != (7,) or self._assembly_offsets.shape != (7,):
            raise ValueError("joint_signs and assembly_offsets must both have length 7")

        self._has_gripper = bool(config.get("gripper", False))
        self._gripper_mode = gripper_mode
        if gripper_output not in {"franka_hand_width", "robotiq_position"}:
            raise ValueError(
                "gripper_output must be 'franka_hand_width' or 'robotiq_position'"
            )
        self._gripper_output = gripper_output
        self._gripper_max_width = float(gripper_max_width)
        self._binary_gripper_threshold = float(binary_gripper_threshold)
        self._last_gripper = 1.0
        self._last_gripper_percent = 1.0
        self._gripper_range_rad = np.asarray(
            config.get("gripper_range_rad", [0.0, 1.0]), dtype=np.float64
        )
        if self._has_gripper and self._gripper_range_rad.shape != (2,):
            raise ValueError("gripper_range_rad must have length 2 when gripper is true")

        self._port = _resolve_port(port or config["com_port"])
        self._joint_ids = list(range(1, 8 + int(self._has_gripper)))
        self._driver = DynamixelDriver(
            self._joint_ids,
            port=self._port,
            baudrate=57600,
            use_fake_fallback=False,
        )
        self._driver.set_torque_mode(False)

        self._prev_arm_raw: np.ndarray | None = None
        self._prev_arm: np.ndarray | None = None
        self._initialize_continuity()
        logger.info("GelloTeleop started on %s", self._port)

    def _initialize_continuity(self) -> None:
        raw = self._read_raw()
        arm_raw = raw[:7]
        arm = _normalize_joint_positions(
            arm_raw, self._assembly_offsets, self._joint_signs
        )
        self._prev_arm_raw = arm_raw.copy()
        self._prev_arm = arm.copy()

    def _read_raw(self) -> np.ndarray:
        raw = np.asarray(self._driver.get_joints(), dtype=np.float64)
        if raw.shape[0] < len(self._joint_ids):
            raise RuntimeError(
                f"Expected at least {len(self._joint_ids)} GELLO joints, got {raw.shape[0]}"
            )
        return raw

    def _read_arm(self, arm_raw: np.ndarray) -> np.ndarray:
        if self._prev_arm_raw is None or self._prev_arm is None:
            self._initialize_continuity()
            assert self._prev_arm_raw is not None
            assert self._prev_arm is not None

        delta = (arm_raw - self._prev_arm_raw) * self._joint_signs
        arm = self._prev_arm + delta
        arm = np.clip(arm, FR3_JOINT_LIMITS[:, 0], FR3_JOINT_LIMITS[:, 1])

        self._prev_arm_raw = arm_raw.copy()
        self._prev_arm = arm.copy()
        return arm

    def _read_gripper(self, raw: np.ndarray) -> tuple[float, float]:
        if not self._has_gripper or raw.shape[0] <= 7:
            return self._last_gripper, self._last_gripper_percent

        low, high = self._gripper_range_rad
        if np.isclose(high, low):
            return self._last_gripper, self._last_gripper_percent

        gripper_percent = float(np.clip((raw[7] - low) / (high - low), 0.0, 1.0))
        if self._gripper_mode == "binary":
            gripper = (
                1.0 if gripper_percent >= self._binary_gripper_threshold else 0.0
            )
        elif self._gripper_mode == "continuous":
            if self._gripper_output == "robotiq_position":
                gripper = (1.0 - gripper_percent) * 255.0
            else:
                gripper = gripper_percent * self._gripper_max_width
        else:
            gripper = gripper_percent
        self._last_gripper = gripper
        self._last_gripper_percent = gripper_percent
        return gripper, gripper_percent

    def get_action(self) -> tuple[np.ndarray, dict]:
        """Return ``[q0..q6, gripper]`` or ``[q0..q6]`` absolute joint action."""
        raw = self._read_raw()
        arm = self._read_arm(raw[:7])
        gripper, gripper_percent = self._read_gripper(raw)

        if self._gripper_mode is None:
            action = arm
        else:
            action = np.append(arm, gripper)

        return action.astype(np.float64), {
            "intervened": True,
            "joint_pos": arm.copy(),
            "gripper": None if self._gripper_mode is None else gripper,
            "gripper_percent": gripper_percent,
            "gripper_output": self._gripper_output,
        }

    def maybe_override(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """GELLO is always authoritative while active."""
        return self.get_action()

    def close(self) -> None:
        """Close the Dynamixel driver."""
        try:
            self._driver.set_torque_mode(False)
        except Exception as exc:
            logger.warning("Failed to disable GELLO torque: %s", exc)
        try:
            self._driver.close()
        except Exception as exc:
            logger.warning("Failed to close GELLO driver: %s", exc)
        logger.info("GelloTeleop stopped.")

    @property
    def action_dim(self) -> int:
        return 8 if self._gripper_mode else 7
