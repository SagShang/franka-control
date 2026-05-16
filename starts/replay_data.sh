python -m franka_control.scripts.replay_lerobot_episode \
  --robot-ip 127.0.0.1 \
  --gripper-host 127.0.0.1 \
  --gripper-type robotiq \
  --repo-id franka/pick_and_place_cube \
  --root data/pick_and_place_cube \
  --episode-index 3 \
  --execute
