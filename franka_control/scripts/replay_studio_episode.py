"""Replay a Franka Data Studio episode on the real robot.

This script replays joint positions from ``samples.jsonl`` and triggers the
Robotiq gripper from ``gripper_action`` transitions.

Expected episode format:
    metadata.json
    samples.jsonl
    wrist_rgb.mp4
    base_rgb.mp4

Gripper action convention defaults to the data-studio convention:
    0 = open, 1 = close
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from franka_control.robot.robot_client import RobotClient
from franka_control.robotiq.robotiq_client import RobotiqClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StudioSample:
    timestamp: float
    frame_index: int
    qpos: np.ndarray
    gripper_action: int | None


@dataclass(frozen=True)
class StudioEpisode:
    root: Path
    metadata: dict
    samples: list[StudioSample]


def _resolve_paths(path: str | Path) -> tuple[Path, Path, Path | None]:
    path = Path(path).expanduser()
    if path.is_dir():
        root = path
        samples_path = root / "samples.jsonl"
        metadata_path = root / "metadata.json"
    else:
        samples_path = path
        root = path.parent
        metadata_path = root / "metadata.json"
    if not samples_path.exists():
        raise FileNotFoundError(f"samples.jsonl not found: {samples_path}")
    return root, samples_path, metadata_path if metadata_path.exists() else None


def load_episode(
    path: str | Path,
    start_frame: int | None = None,
    end_frame: int | None = None,
    stride: int = 1,
) -> StudioEpisode:
    if stride < 1:
        raise ValueError("stride must be >= 1")

    root, samples_path, metadata_path = _resolve_paths(path)
    metadata = {}
    if metadata_path is not None:
        metadata = json.loads(metadata_path.read_text())

    samples: list[StudioSample] = []
    with samples_path.open() as f:
        for line in f:
            row = json.loads(line)
            frame_index = int(row["frame_index"])
            if start_frame is not None and frame_index < start_frame:
                continue
            if end_frame is not None and frame_index > end_frame:
                continue
            if (frame_index - (start_frame or 0)) % stride != 0:
                continue

            qpos = np.asarray(row["robot_state"]["position"], dtype=np.float64)
            if qpos.shape != (7,):
                raise ValueError(
                    f"frame {frame_index}: expected 7 joint positions, got {qpos.shape}"
                )
            action = row.get("gripper_action")
            samples.append(StudioSample(
                timestamp=float(row["timestamp"]),
                frame_index=frame_index,
                qpos=qpos,
                gripper_action=None if action is None else int(action),
            ))

    if not samples:
        raise ValueError("No samples selected")
    return StudioEpisode(root=root, metadata=metadata, samples=samples)


def gripper_transitions(samples: Iterable[StudioSample]) -> list[StudioSample]:
    transitions = []
    previous = None
    for sample in samples:
        action = sample.gripper_action
        if action is None:
            continue
        if previous is None or action != previous:
            transitions.append(sample)
        previous = action
    return transitions


def _action_name(action: int, open_action: int, close_action: int) -> str | None:
    if action == open_action:
        return "open"
    if action == close_action:
        return "close"
    return None


def summarize_episode(
    episode: StudioEpisode,
    open_action: int,
    close_action: int,
) -> None:
    samples = episode.samples
    first = samples[0]
    last = samples[-1]
    duration = last.timestamp - first.timestamp
    q = np.stack([s.qpos for s in samples])
    logger.info("Episode: %s", episode.root)
    logger.info("Samples: %d | frames: %d..%d | duration: %.3f s",
                len(samples), first.frame_index, last.frame_index, duration)
    if episode.metadata:
        logger.info("Format: %s | prompt: %s",
                    episode.metadata.get("format", "?"),
                    episode.metadata.get("prompt", ""))
    logger.info("First qpos: %s", np.round(first.qpos, 4).tolist())
    logger.info("Last qpos:  %s", np.round(last.qpos, 4).tolist())
    logger.info("Joint spans: %s", np.round(q.max(axis=0) - q.min(axis=0), 4).tolist())

    transitions = gripper_transitions(samples)
    if transitions:
        logger.info(
            "Gripper mapping: action %d=open, action %d=close",
            open_action,
            close_action,
        )
        for sample in transitions:
            name = _action_name(sample.gripper_action, open_action, close_action)
            logger.info("Gripper event: frame=%d t=%.3f action=%s",
                        sample.frame_index, sample.timestamp - first.timestamp,
                        name or sample.gripper_action)
    else:
        logger.info("No gripper_action transitions found.")


def _confirm_execute(args, episode: StudioEpisode) -> None:
    if args.yes:
        return
    print()
    print("About to replay on the real robot:")
    print(f"  episode: {episode.root}")
    print(f"  robot:   {args.robot_ip}:{args.robot_port}")
    if not args.no_gripper:
        print(f"  gripper: {args.gripper_host}:{args.gripper_port}")
        print(f"  mapping: {args.open_action}=open, {args.close_action}=close")
    print("Type REPLAY to continue: ", end="", flush=True)
    if input().strip() != "REPLAY":
        raise SystemExit("Aborted.")


def _apply_gripper_action(
    gripper: RobotiqClient,
    action: int,
    open_action: int,
    close_action: int,
    speed: int,
    force: int,
    wait: bool,
) -> str | None:
    name = _action_name(action, open_action, close_action)
    if name == "open":
        ok = gripper.open(speed=speed, force=force, wait=wait)
    elif name == "close":
        ok = gripper.close(speed=speed, force=force, wait=wait)
    else:
        logger.warning("Ignoring unknown gripper_action=%s", action)
        return None
    if not ok:
        raise RuntimeError(f"Robotiq {name} command failed")
    return name


def replay_episode(args, episode: StudioEpisode) -> None:
    if not args.robot_ip:
        raise ValueError("--robot-ip is required with --execute")

    gripper_host = args.gripper_host or args.robot_ip
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

        if not args.no_gripper:
            gripper = RobotiqClient(gripper_host, port=args.gripper_port)
            logger.info("Activating Robotiq without reset...")
            if not gripper.activate(reset=False, start=True, open_after=False):
                raise RuntimeError("Robotiq activate failed")

        first = episode.samples[0]
        if not args.skip_initial_move:
            logger.info("Moving to first recorded qpos...")
            if not robot.move(first.qpos, timeout=args.move_timeout):
                raise RuntimeError("Initial robot.move() failed")

        _confirm_execute(args, episode)

        last_action = None
        if gripper is not None and args.apply_initial_gripper:
            if first.gripper_action is not None:
                event = _apply_gripper_action(
                    gripper,
                    first.gripper_action,
                    args.open_action,
                    args.close_action,
                    args.gripper_speed,
                    args.gripper_force,
                    wait=True,
                )
                last_action = first.gripper_action
                if event:
                    logger.info("Initial gripper: %s", event)

        base_ts = first.timestamp
        wall_t0 = time.perf_counter()
        sent = 0
        for sample in episode.samples:
            target_elapsed = (sample.timestamp - base_ts) * args.time_scale
            while True:
                remaining = target_elapsed - (time.perf_counter() - wall_t0)
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.002))

            if (
                gripper is not None
                and sample.gripper_action is not None
                and sample.gripper_action != last_action
            ):
                event = _apply_gripper_action(
                    gripper,
                    sample.gripper_action,
                    args.open_action,
                    args.close_action,
                    args.gripper_speed,
                    args.gripper_force,
                    wait=args.gripper_wait,
                )
                last_action = sample.gripper_action
                if event:
                    logger.info("Gripper %s at frame %d t=%.3f",
                                event, sample.frame_index,
                                sample.timestamp - base_ts)

            if not robot.set("q_desired", sample.qpos):
                raise RuntimeError(f"Failed to send q_desired at frame {sample.frame_index}")
            sent += 1
            if args.log_every > 0 and sent % args.log_every == 0:
                logger.info("Sent %d/%d frames (frame=%d)",
                            sent, len(episode.samples), sample.frame_index)

        logger.info("Replay complete. Sent %d frames.", sent)
    finally:
        if gripper is not None:
            gripper.disconnect()
        robot.close()


def main():
    parser = argparse.ArgumentParser(
        description="Replay a franka_data_studio episode using joint angles."
    )
    parser.add_argument("episode", help="Episode directory or samples.jsonl path")
    parser.add_argument("--robot-ip", default=None, help="Robot server host")
    parser.add_argument("--robot-port", type=int, default=5555)
    parser.add_argument("--state-stream-port", type=int, default=5557)
    parser.add_argument("--gripper-host", default=None,
                        help="Robotiq server host (default: --robot-ip)")
    parser.add_argument("--gripper-port", type=int, default=5556)
    parser.add_argument("--no-gripper", action="store_true",
                        help="Replay robot joints only")
    parser.add_argument("--execute", action="store_true",
                        help="Actually command the real robot")
    parser.add_argument("--yes", action="store_true",
                        help="Skip interactive REPLAY confirmation")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help=">1 slower, <1 faster (default: 1.0)")
    parser.add_argument("--move-timeout", type=float, default=30.0)
    parser.add_argument("--skip-initial-move", action="store_true",
                        help="Do not move to the first recorded qpos before replay")
    parser.add_argument("--open-action", type=int, default=0)
    parser.add_argument("--close-action", type=int, default=1)
    parser.add_argument("--apply-initial-gripper", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--gripper-speed", type=int, default=255)
    parser.add_argument("--gripper-force", type=int, default=128)
    parser.add_argument("--gripper-wait", action="store_true",
                        help="Wait for Robotiq motion at action transitions")
    parser.add_argument("--log-every", type=int, default=120)
    args = parser.parse_args()

    if args.time_scale <= 0:
        parser.error("--time-scale must be > 0")

    episode = load_episode(
        args.episode,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
    )
    summarize_episode(episode, args.open_action, args.close_action)

    if not args.execute:
        logger.info("Dry run only. Add --execute to command the real robot.")
        return

    replay_episode(args, episode)


if __name__ == "__main__":
    main()
