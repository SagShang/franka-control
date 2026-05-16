"""Run a Franka policy loop against an OpenPI websocket server."""

from __future__ import annotations

import argparse
import logging
import signal
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from franka_control.openpi import ActionChunkQueue, OpenPIPolicyClient
from franka_control.openpi.adapter import (
    OpenPIActionConfig,
    OpenPIObservationConfig,
    make_openpi_observation,
    openpi_action_to_env_action,
)

if TYPE_CHECKING:
    from franka_control.cameras import CameraManager
    from franka_control.envs import FrankaEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_image_size(value: str | None) -> tuple[int, int] | None:
    if value is None or value == "":
        return None
    if "x" in value.lower():
        height, width = value.lower().split("x", 1)
        return int(height), int(width)
    size = int(value)
    return size, size


def _confirm_execute(args: argparse.Namespace) -> None:
    if args.dry_run or args.yes:
        return
    print()
    print("About to run OpenPI inference on the real robot:")
    print(f"  robot:   {args.robot_ip}:{args.robot_port}")
    if not args.no_gripper:
        print(f"  gripper: {(args.gripper_host or args.robot_ip)}:{args.gripper_port}")
    print(f"  server:  {args.server_scheme}://{args.host}:{args.port}")
    print(f"  prompt:  {args.prompt}")
    print(f"  mode:    {args.control_mode} @ {args.hz:.1f} Hz")
    print("Type RUN to continue: ", end="", flush=True)
    if input().strip() != "RUN":
        raise SystemExit("Aborted.")


def _camera_rgb_frames(cameras: "CameraManager") -> dict[str, np.ndarray]:
    raw = cameras.read_latest()
    images = {
        name: data["rgb"].copy()
        for name, data in raw.items()
        if "rgb" in data
    }
    return images


def _wait_for_camera_images(
    cameras: "CameraManager",
    required: set[str],
    timeout: float,
) -> dict[str, np.ndarray]:
    deadline = time.monotonic() + timeout
    last_images: dict[str, np.ndarray] = {}
    while time.monotonic() < deadline:
        images = _camera_rgb_frames(cameras)
        if images:
            last_images.update(images)
        if required.issubset(last_images):
            return last_images
        time.sleep(0.02)
    missing = sorted(required - set(last_images))
    raise RuntimeError(f"Timed out waiting for camera images: {missing}")


def _make_observation(
    env: "FrankaEnv",
    cameras: "CameraManager",
    last_images: dict[str, np.ndarray],
    obs_config: OpenPIObservationConfig,
) -> tuple[dict, dict[str, np.ndarray]]:
    robot_obs = env.get_observation()
    images = _camera_rgb_frames(cameras)
    if images:
        last_images.update(images)
    return make_openpi_observation(robot_obs, last_images, obs_config), last_images


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Franka remote inference loop against an OpenPI server."
    )

    # OpenPI server
    parser.add_argument("--host", required=True, help="OpenPI policy server host")
    parser.add_argument("--port", type=int, default=8001, help="OpenPI server port")
    parser.add_argument("--api-key", default=None, help="Optional OpenPI API key")
    parser.add_argument(
        "--server-scheme",
        default="ws",
        choices=["ws", "wss"],
        help="OpenPI websocket scheme",
    )
    parser.add_argument(
        "--action-key",
        default="actions",
        help="Policy response key containing the action chunk",
    )
    parser.add_argument(
        "--return-horizon",
        type=int,
        default=16,
        help="Expected returned action chunk horizon",
    )

    # Robot and gripper
    parser.add_argument("--robot-ip", required=True, help="Control machine IP")
    parser.add_argument("--robot-port", type=int, default=5555)
    parser.add_argument("--state-stream-port", type=int, default=5557)
    parser.add_argument("--gripper-host", default=None)
    parser.add_argument("--gripper-port", type=int, default=5556)
    parser.add_argument(
        "--gripper-type",
        choices=["franka_hand", "robotiq"],
        default="robotiq",
    )
    parser.add_argument(
        "--gripper-mode",
        choices=["binary", "continuous"],
        default="binary",
    )
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument(
        "--control-mode",
        choices=["joint_abs", "joint_delta", "ee_abs", "ee_delta"],
        default="joint_abs",
        help="FrankaEnv action mode used to apply policy actions",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Move to FrankaEnv home qpos before starting inference",
    )

    # Observation
    parser.add_argument("--prompt", required=True, help="Language instruction")
    parser.add_argument(
        "--cameras",
        type=Path,
        default=Path("config/cameras.yaml"),
        help="Camera YAML path",
    )
    parser.add_argument(
        "--high-camera",
        default="camera_2",
        help="CameraManager name mapped to OpenPI images.cam_high",
    )
    parser.add_argument(
        "--wrist-camera",
        default="d435i",
        help="CameraManager name mapped to OpenPI images.cam_wrist",
    )
    parser.add_argument(
        "--image-size",
        default=None,
        help="Optional HxW resize, or one integer for square resize",
    )
    parser.add_argument(
        "--camera-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for initial camera frames",
    )
    parser.add_argument(
        "--gripper-close-width",
        type=float,
        default=0.01,
        help="Franka Hand width threshold mapped to closed state",
    )
    parser.add_argument(
        "--robotiq-close-position",
        type=float,
        default=128.0,
        help="Robotiq position threshold mapped to closed state",
    )
    parser.add_argument(
        "--invert-gripper-state",
        action="store_true",
        help="Invert OpenPI gripper value in observation state",
    )
    parser.add_argument(
        "--no-invert-gripper-action",
        action="store_true",
        help="Do not map OpenPI 0=open/1=closed into FrankaEnv binary semantics",
    )

    # Loop control
    parser.add_argument("--hz", type=float, default=20.0, help="Control frequency")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run forever")
    parser.add_argument("--dry-run", action="store_true", help="Infer but do not step robot")
    parser.add_argument("--yes", action="store_true", help="Skip real-robot confirmation")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    from franka_control.cameras import CameraManager
    from franka_control.envs import FrankaEnv

    if args.hz <= 0.0:
        raise ValueError("--hz must be positive")
    period = 1.0 / args.hz

    obs_config = OpenPIObservationConfig(
        high_camera=args.high_camera,
        wrist_camera=args.wrist_camera,
        prompt=args.prompt,
        image_size=_parse_image_size(args.image_size),
        gripper_type=args.gripper_type,
        gripper_close_width=args.gripper_close_width,
        robotiq_close_position=args.robotiq_close_position,
        gripper_invert_state=args.invert_gripper_state,
    )
    invert_gripper_action = (
        args.gripper_mode == "binary" or args.gripper_type == "franka_hand"
    )
    if args.no_invert_gripper_action:
        invert_gripper_action = False
    action_config = OpenPIActionConfig(
        mode=args.control_mode,
        include_gripper=not args.no_gripper,
        binary_gripper=args.gripper_mode == "binary",
        gripper_invert=invert_gripper_action,
    )

    env = FrankaEnv(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        state_stream_port=args.state_stream_port,
        gripper_host=None if args.no_gripper else (args.gripper_host or args.robot_ip),
        gripper_port=args.gripper_port,
        gripper_type=args.gripper_type,
        gripper_mode=args.gripper_mode,
        action_mode=args.control_mode,
    )
    cameras = None
    policy = None
    queue = ActionChunkQueue(
        action_key=args.action_key,
        return_horizon=args.return_horizon,
    )
    running = True

    def _signal_handler(sig, frame):
        nonlocal running
        running = False
        logger.info("Stopping OpenPI inference loop...")

    signal.signal(signal.SIGINT, _signal_handler)

    try:
        logger.info("Starting cameras from %s", args.cameras)
        cameras = CameraManager.from_yaml(args.cameras)
        required_cameras = {args.high_camera, args.wrist_camera}
        last_images = _wait_for_camera_images(
            cameras,
            required=required_cameras,
            timeout=args.camera_timeout,
        )
        logger.info("Camera frames ready: %s", sorted(last_images))

        logger.info("Connecting robot environment...")
        if args.reset:
            env.reset()
        else:
            env.connect()

        policy = OpenPIPolicyClient(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            scheme=args.server_scheme,
        )
        queue.reset()
        _confirm_execute(args)

        logger.info("Starting OpenPI inference loop")
        step = 0
        while running and (args.max_steps <= 0 or step < args.max_steps):
            loop_start = time.monotonic()
            observation, last_images = _make_observation(
                env,
                cameras,
                last_images,
                obs_config,
            )
            policy_action = queue.next_action(policy, observation)
            env_action = openpi_action_to_env_action(
                policy_action,
                env.action_space.low,
                env.action_space.high,
                action_config,
            )

            if args.dry_run:
                logger.info(
                    "Dry-run action[%d]: policy=%s env=%s",
                    step,
                    np.array2string(policy_action, precision=4),
                    np.array2string(env_action, precision=4),
                )
            else:
                env.step(env_action)

            step = queue.mark_action_executed()
            elapsed = time.monotonic() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
            else:
                logger.warning(
                    "Loop overran period %.3fs with elapsed %.3fs; queue_len=%d",
                    period,
                    elapsed,
                    queue.queue_length,
                )

        logger.info("OpenPI inference finished after %d steps", step)
    finally:
        queue.close()
        if policy is not None:
            policy.close()
        if cameras is not None:
            cameras.close()
        env.close()


if __name__ == "__main__":
    main()
