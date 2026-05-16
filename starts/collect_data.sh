python -m franka_control.scripts.collect_episodes \
  --robot-ip 127.0.0.1 \
  --gripper-host 127.0.0.1 \
  --gripper-type robotiq \
  --repo-id test/gello_with_camera \
  --root data/gello_with_camera \
  --task-name "gello validation" \
  --device gello \
  --control-mode joint_abs \
  --gello-root /home/shang/gello_software \
  --gello-config /home/shang/gello_software/ros2/src/franka_gello_state_publisher/config/franka_gello_single.yaml \
  --fps 30 \
  --num-episodes 1 \
  --cameras config/cameras.yaml \
  --display off
  
