python -m franka_control.scripts.teleop \
  --robot-ip 127.0.0.1 \
  --gripper-host 127.0.0.1 \
  --gripper-type robotiq \
  --device gello \
  --gello-root /home/shang/gello_software \
  --gello-config /home/shang/gello_software/ros2/src/franka_gello_state_publisher/config/franka_gello_single.yaml \
  --gello-section SINGLE \
  --hz 50
