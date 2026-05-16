"""Replay a LeRobot joint-action episode on the real Franka.

This replays ``action`` from a collected LeRobot dataset:
    action[:7] -> Franka joint_abs q_desired [rad]
    action[7]  -> gripper target

For Robotiq datasets, the gripper target is the native 0..255 position where
0 is open and 255 is closed.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from franka_control.envs.franka_env import JOINT_LIMIT_HIGH, JOINT_LIMIT_LOW
from franka_control.gripper.gripper_client import GripperClient
from franka_control.robot.robot_client import DEFAULT_STATE_STREAM_PORT, RobotClient
from franka_control.robotiq.robotiq_client import RobotiqClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReplayFrame:
    frame_index: int
    timestamp: float
    action: np.ndarray


@dataclass(frozen=True)
class ReplayEpisode:
    root: Path
    repo_id: str
    episode_index: int
    fps: int
    frames: list[ReplayFrame]
    task: str


def _read_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    return json.loads(info_path.read_text())


def _episode_task(root: Path, episode_index: int) -> str:
    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    for path in episode_files:
        table = pd.read_parquet(path)
        if "episode_index" not in table.columns:
            continue
        matches = table[table["episode_index"] == episode_index]
        if matches.empty or "tasks" not in matches.columns:
            continue
        tasks = matches.iloc[0]["tasks"]
        if isinstance(tasks, np.ndarray):
            tasks = tasks.tolist()
        if isinstance(tasks, list):
            return ", ".join(str(task) for task in tasks)
        return str(tasks)
    return ""


def _read_dataset_frames(root: Path, episode_index: int) -> pd.DataFrame:
    data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet data files found under {root / 'data'}")

    frames = []
    for path in data_files:
        table = pd.read_parquet(path)
        if "episode_index" not in table.columns:
            raise ValueError(f"{path} has no episode_index column")
        selected = table[table["episode_index"] == episode_index]
        if not selected.empty:
            frames.append(selected)

    if not frames:
        raise ValueError(f"No frames found for episode_index={episode_index}")
    return pd.concat(frames, ignore_index=True).sort_values("frame_index")


def load_episode(
    root: str | Path,
    repo_id: str,
    episode_index: int = 0,
    start_frame: int | None = None,
    end_frame: int | None = None,
    stride: int = 1,
) -> ReplayEpisode:
    """Load one LeRobot episode from local parquet files."""
    if stride < 1:
        raise ValueError("stride must be >= 1")

    root = Path(root).expanduser()
    info = _read_info(root)
    table = _read_dataset_frames(root, episode_index)

    if start_frame is not None:
        table = table[table["frame_index"] >= start_frame]
    if end_frame is not None:
        table = table[table["frame_index"] <= end_frame]
    if stride > 1:
        base = int(start_frame if start_frame is not None else table["frame_index"].min())
        table = table[((table["frame_index"] - base) % stride) == 0]
    if table.empty:
        raise ValueError("No frames selected")

    replay_frames: list[ReplayFrame] = []
    for _, row in table.iterrows():
        frame_index = int(row["frame_index"])
        timestamp = float(row["timestamp"])
        action = np.asarray(row["action"], dtype=np.float64)
        if action.shape[0] < 7:
            raise ValueError(
                f"frame {frame_index}: expected action with at least 7 values, got {action.shape}"
            )
        replay_frames.append(
            ReplayFrame(
                frame_index=frame_index,
                timestamp=timestamp,
                action=action,
            )
        )

    return ReplayEpisode(
        root=root,
        repo_id=repo_id,
        episode_index=episode_index,
        fps=int(info.get("fps", 0)),
        frames=replay_frames,
        task=_episode_task(root, episode_index),
    )


def summarize_episode(episode: ReplayEpisode) -> None:
    frames = episode.frames
    first = frames[0]
    last = frames[-1]
    actions = np.stack([frame.action for frame in frames])
    joints = actions[:, :7]
    duration = last.timestamp - first.timestamp

    logger.info("Dataset root: %s", episode.root)
    logger.info("Repo id: %s | episode: %d | task: %s",
                episode.repo_id, episode.episode_index, episode.task or "?")
    logger.info(
        "Frames: %d | frame_index: %d..%d | duration: %.3f s | fps(meta): %s",
        len(frames),
        first.frame_index,
        last.frame_index,
        duration,
        episode.fps or "?",
    )
    logger.info("First q: %s", np.round(first.action[:7], 4).tolist())
    logger.info("Last q:  %s", np.round(last.action[:7], 4).tolist())
    logger.info("Joint span: %s", np.round(joints.max(axis=0) - joints.min(axis=0), 4).tolist())

    out_of_bounds = np.logical_or(joints < JOINT_LIMIT_LOW, joints > JOINT_LIMIT_HIGH)
    if np.any(out_of_bounds):
        bad = np.argwhere(out_of_bounds)
        frame, joint = bad[0]
        raise ValueError(
            "Selected episode contains joint targets outside FR3 limits: "
            f"frame={frames[int(frame)].frame_index}, joint={int(joint)}, "
            f"value={joints[int(frame), int(joint)]:.6f}"
        )

    if actions.shape[1] > 7:
        gripper = actions[:, 7]
        logger.info(
            "Gripper action: min=%.3f max=%.3f mean=%.3f",
            float(gripper.min()),
            float(gripper.max()),
            float(gripper.mean()),
        )


def _confirm_execute(args: argparse.Namespace, episode: ReplayEpisode) -> None:
    if args.yes:
        return
    print()
    print("About to replay this LeRobot episode on the real robot:")
    print(f"  dataset: {episode.root}")
    print(f"  episode: {episode.episode_index}")
    print(f"  frames:  {len(episode.frames)}")
    print(f"  robot:   {args.robot_ip}:{args.robot_port}")
    if not args.no_gripper:
        print(f"  gripper: {args.gripper_type} at {args.gripper_host or args.robot_ip}:{args.gripper_port}")
    print("Type REPLAY to continue: ", end="", flush=True)
    if input().strip() != "REPLAY":
        raise SystemExit("Aborted.")


def _sleep_until(target_elapsed: float, wall_t0: float) -> None:
    while True:
        remaining = target_elapsed - (time.perf_counter() - wall_t0)
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.002))


def _connect_gripper(args: argparse.Namespace):
    if args.no_gripper:
        return None

    host = args.gripper_host or args.robot_ip
    if args.gripper_type == "robotiq":
        gripper = RobotiqClient(host, port=args.gripper_port)
        if not gripper.activate(reset=False, start=True, open_after=False):
            raise RuntimeError("Robotiq activate failed")
        return gripper

    if args.gripper_type == "franka_hand":
        gripper = GripperClient(host, port=args.gripper_port)
        return gripper

    raise ValueError(f"Unsupported gripper type: {args.gripper_type}")


def _apply_gripper(args: argparse.Namespace, gripper, value: float) -> bool:
    if gripper is None:
        return True

    if args.gripper_type == "robotiq":
        position = int(round(float(np.clip(value, 0.0, 255.0))))
        return bool(gripper.move(
            position=position,
            speed=args.robotiq_speed,
            force=args.robotiq_force,
            wait=False,
        ))

    width = float(np.clip(value, 0.0, args.franka_hand_max_width))
    return bool(gripper.move(width=width, speed=args.franka_hand_speed))


def _disconnect_gripper(args: argparse.Namespace, gripper) -> None:
    if gripper is None:
        return
    if args.gripper_type == "robotiq":
        gripper.disconnect()
    else:
        gripper.close()


def replay_episode(args: argparse.Namespace, episode: ReplayEpisode) -> None:
    robot = RobotClient(
        host=args.robot_ip,
        port=args.robot_port,
        state_stream_port=args.state_stream_port,
    )
    gripper = None
    try:
        logger.info("Connecting robot in PID mode...")
        if not robot.connect(controller_type="pid"):
            raise RuntimeError("Robot connect failed")
        if not robot.switch("pid"):
            raise RuntimeError("Robot switch(pid) failed")

        gripper = _connect_gripper(args)

        first = episode.frames[0]
        _confirm_execute(args, episode)

        if not args.skip_initial_move:
            logger.info("Moving to first recorded qpos...")
            if not robot.move(first.action[:7], timeout=args.move_timeout):
                raise RuntimeError("Initial robot.move() failed")

        base_ts = first.timestamp
        wall_t0 = time.perf_counter()
        sent = 0
        gripper_rejected = 0

        for frame in episode.frames:
            target_elapsed = (frame.timestamp - base_ts) * args.time_scale
            _sleep_until(target_elapsed, wall_t0)

            if frame.action.shape[0] > 7 and gripper is not None:
                if not _apply_gripper(args, gripper, float(frame.action[7])):
                    gripper_rejected += 1

            if not robot.set("q_desired", frame.action[:7]):
                raise RuntimeError(
                    f"Failed to send q_desired at frame {frame.frame_index}"
                )

            sent += 1
            if args.log_every > 0 and sent % args.log_every == 0:
                logger.info(
                    "Sent %d/%d frames (frame=%d)",
                    sent,
                    len(episode.frames),
                    frame.frame_index,
                )

        logger.info(
            "Replay complete. Sent %d frames. Gripper rejected commands: %d",
            sent,
            gripper_rejected,
        )
    finally:
        _disconnect_gripper(args, gripper)
        robot.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a LeRobot joint_abs episode on the real Franka."
    )
    parser.add_argument("--repo-id", required=True, help="Dataset repo id")
    parser.add_argument("--root", required=True, help="Local dataset root")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--robot-ip", required=True)
    parser.add_argument("--robot-port", type=int, default=5555)
    parser.add_argument("--state-stream-port", type=int, default=DEFAULT_STATE_STREAM_PORT)
    parser.add_argument("--gripper-host", default=None)
    parser.add_argument("--gripper-port", type=int, default=5556)
    parser.add_argument("--gripper-type", choices=["robotiq", "franka_hand"], default="robotiq")
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--execute", action="store_true",
                        help="Actually command the real robot")
    parser.add_argument("--yes", action="store_true",
                        help="Skip interactive REPLAY confirmation")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help=">1 slower, <1 faster")
    parser.add_argument("--move-timeout", type=float, default=30.0)
    parser.add_argument("--skip-initial-move", action="store_true")
    parser.add_argument("--robotiq-speed", type=int, default=255)
    parser.add_argument("--robotiq-force", type=int, default=128)
    parser.add_argument("--franka-hand-max-width", type=float, default=0.08)
    parser.add_argument("--franka-hand-speed", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=120)
    args = parser.parse_args()

    if args.time_scale <= 0:
        parser.error("--time-scale must be > 0")

    episode = load_episode(
        root=args.root,
        repo_id=args.repo_id,
        episode_index=args.episode_index,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
    )
    summarize_episode(episode)

    if not args.execute:
        logger.info("Dry run only. Add --execute to command the real robot.")
        return

    replay_episode(args, episode)


if __name__ == "__main__":
    main()
