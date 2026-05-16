python -m franka_control.scripts.replay_lerobot_episode \
  --robot-ip 127.0.0.1 \
  --gripper-host 127.0.0.1 \
  --gripper-type robotiq \
  --repo-id test/gello_with_camera \
  --root data/gello_with_camera \
  --episode-index 0 \
  --execute
