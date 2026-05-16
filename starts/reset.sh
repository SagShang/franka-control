#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 - "$@" <<'PY'
import argparse
import logging
import sys

TARGET_QPOS = [
    1.5811532256528844,
    0.3352566519215481,
    -0.039126062352400126,
    -1.421400349829057,
    0.019179442329607888,
    1.7957830641030057,
    -0.02342267720573643,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="./reset.sh",
        description=(
            "Move the Franka arm to the reset joint pose, then optionally "
            "open or close a selected gripper."
        ),
        epilog=(
            "Examples:\n"
            "  ./reset.sh\n"
            "  ./reset.sh franka_hand open --robot-ip 192.168.0.100\n"
            "  ./reset.sh robotiq close --robot-ip 192.168.0.100\n"
            "\n"
            "Defaults: --robot-ip 127.0.0.1 --gripper-type robotiq "
            "--gripper-action open"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "positional_gripper_type",
        nargs="?",
        choices=("none", "franka_hand", "robotiq"),
        help="Optional shorthand for --gripper-type.",
    )
    parser.add_argument(
        "positional_gripper_action",
        nargs="?",
        choices=("none", "open", "close"),
        help="Optional shorthand for --gripper-action.",
    )
    parser.add_argument(
        "--robot-ip",
        default="127.0.0.1",
        help="RobotServer host/control-machine IP (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--robot-port",
        type=int,
        default=5555,
        help="RobotServer command port (default: 5555).",
    )
    parser.add_argument(
        "--state-stream-port",
        type=int,
        default=5557,
        help="RobotServer state stream port (default: 5557).",
    )
    parser.add_argument(
        "--move-timeout",
        type=float,
        default=60.0,
        help="Robot connect/move timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--gripper-host",
        default=None,
        help="Gripper server host (default: same as --robot-ip).",
    )
    parser.add_argument(
        "--gripper-port",
        type=int,
        default=5556,
        help="Gripper server command port (default: 5556).",
    )
    parser.add_argument(
        "--gripper-type",
        choices=("none", "franka_hand", "robotiq"),
        default="robotiq",
        help="Gripper type to command after arm reset (default: robotiq).",
    )
    parser.add_argument(
        "--gripper-action",
        choices=("none", "open", "close"),
        default="open",
        help="Gripper action after arm reset (default: open).",
    )
    parser.add_argument(
        "--gripper-timeout",
        type=float,
        default=30.0,
        help="Gripper command timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--franka-open-width",
        type=float,
        default=0.08,
        help="Franka Hand open width in meters (default: 0.08).",
    )
    parser.add_argument(
        "--franka-close-width",
        type=float,
        default=0.0,
        help="Franka Hand close/grasp target width in meters (default: 0.0).",
    )
    parser.add_argument(
        "--franka-speed",
        type=float,
        default=0.1,
        help="Franka Hand speed in m/s (default: 0.1).",
    )
    parser.add_argument(
        "--franka-force",
        type=float,
        default=40.0,
        help="Franka Hand grasp force in N if you adapt close to grasp (default: 40).",
    )
    parser.add_argument(
        "--robotiq-speed",
        type=int,
        default=255,
        help="Robotiq speed register 0..255 (default: 255).",
    )
    parser.add_argument(
        "--robotiq-force",
        type=int,
        default=128,
        help="Robotiq force register 0..255 (default: 128).",
    )
    parser.add_argument(
        "--skip-connect",
        action="store_true",
        help="Skip RobotClient.connect() and only send move to an existing controller.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    args = parser.parse_args()
    if args.positional_gripper_type is not None:
        args.gripper_type = args.positional_gripper_type
    if args.positional_gripper_action is not None:
        args.gripper_action = args.positional_gripper_action
    return args


def reset_robot(args: argparse.Namespace) -> None:
    import numpy as np

    from franka_control.robot.robot_client import RobotClient

    target = np.array(TARGET_QPOS, dtype=np.float64)
    print(f"Connecting to robot server {args.robot_ip}:{args.robot_port} ...")
    with RobotClient(
        host=args.robot_ip,
        port=args.robot_port,
        state_stream_port=args.state_stream_port,
        timeout=args.move_timeout,
    ) as robot:
        if not args.skip_connect:
            print("Starting PID controller ...")
            if not robot.connect(controller_type="pid", timeout=args.move_timeout):
                raise RuntimeError("Failed to connect robot controller")

        print("Moving arm to reset joint pose ...")
        if not robot.move(target, timeout=args.move_timeout):
            raise RuntimeError("Failed to move robot to reset joint pose")
        print("Arm reset complete.")


def command_gripper(args: argparse.Namespace) -> None:
    if args.gripper_type == "none" or args.gripper_action == "none":
        print("No gripper action requested.")
        return

    host = args.gripper_host or args.robot_ip

    if args.gripper_type == "franka_hand":
        from franka_control.gripper.gripper_client import GripperClient

        print(
            f"Connecting to Franka Hand server {host}:{args.gripper_port} ..."
        )
        with GripperClient(
            host=host,
            port=args.gripper_port,
            timeout=args.gripper_timeout,
        ) as gripper:
            if args.gripper_action == "open":
                print("Opening Franka Hand ...")
                ok = gripper.open(
                    width=args.franka_open_width,
                    speed=args.franka_speed,
                    timeout=args.gripper_timeout,
                )
            else:
                print("Closing Franka Hand ...")
                resp = gripper._send_command(
                    "move",
                    {
                        "width": args.franka_close_width,
                        "speed": args.franka_speed,
                    },
                )
                if not resp.get("success"):
                    raise RuntimeError(
                        f"Franka Hand close rejected: {resp.get('error')}"
                    )
                result = gripper._wait_until_idle(args.gripper_timeout)
                ok = result.get("success", False)
        if not ok:
            raise RuntimeError(f"Franka Hand {args.gripper_action} failed")
        print(f"Franka Hand {args.gripper_action} complete.")
        return

    if args.gripper_type == "robotiq":
        from franka_control.robotiq.robotiq_client import RobotiqClient

        print(f"Connecting to Robotiq server {host}:{args.gripper_port} ...")
        with RobotiqClient(
            host=host,
            port=args.gripper_port,
            timeout=args.gripper_timeout,
        ) as gripper:
            print("Activating Robotiq gripper if needed ...")
            if not gripper.activate(
                reset=False,
                start=True,
                open_after=False,
                speed=args.robotiq_speed,
                force=args.robotiq_force,
                timeout=args.gripper_timeout,
            ):
                raise RuntimeError("Robotiq activation failed")

            if args.gripper_action == "open":
                print("Opening Robotiq gripper ...")
                ok = gripper.open(
                    speed=args.robotiq_speed,
                    force=args.robotiq_force,
                    timeout=args.gripper_timeout,
                )
            else:
                print("Closing Robotiq gripper ...")
                ok = gripper.close(
                    speed=args.robotiq_speed,
                    force=args.robotiq_force,
                    timeout=args.gripper_timeout,
                )
        if not ok:
            raise RuntimeError(f"Robotiq {args.gripper_action} failed")
        print(f"Robotiq {args.gripper_action} complete.")
        return

    raise RuntimeError(f"Unsupported gripper type: {args.gripper_type}")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.gripper_type == "none" and args.gripper_action != "none":
        raise ValueError("--gripper-action requires --gripper-type")
    if args.gripper_type != "none" and args.gripper_action == "none":
        raise ValueError("--gripper-type requires --gripper-action open or close")

    reset_robot(args)
    command_gripper(args)
    print("Reset sequence finished.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
PY
