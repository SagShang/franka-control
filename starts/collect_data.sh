python -m franka_control.scripts.collect_episodes \
  --robot-ip 127.0.0.1 \
  --gripper-host 127.0.0.1 \
  --gripper-type robotiq \
  --repo-id franka/pick_and_place_cube \
  --root data/pick_and_place_cube \
  --task-name "pick up the cube and place it in the basket" \
  --device gello \
  --control-mode joint_abs \
  --gello-root /home/shang/gello_software \
  --gello-config /home/shang/gello_software/ros2/src/franka_gello_state_publisher/config/franka_gello_single.yaml \
  --fps 30 \
  --num-episodes 50 \
  --cameras config/cameras.yaml \
  --display on \
  "$@"
