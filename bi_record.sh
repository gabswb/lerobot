#!/bin/bash

lerobot-record \
  --robot.type=bi_so101_follower \
  --robot.right_arm_port=/dev/f1 \
  --robot.left_arm_port=/dev/f0 \
  --robot.right_arm_id=f1 \
  --robot.left_arm_id=f0 \
  --teleop.type=bi_so101_leader \
  --teleop.left_arm_port=/dev/l0 \
  --teleop.right_arm_port=/dev/l1 \
  --teleop.left_arm_id=l0 \
  --teleop.right_arm_id=l1 \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 8, "width": 640, "height": 480, "fps": 30},
    scanner: {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 30}
  }' \
  --display_data=true \
  --dataset.repo_id="tms-gvd/scan-v2-cycling" \
  --dataset.num_episodes=50 \
  --dataset.single_task="Take an item, scan it then move it to the corresponding area" \
  --dataset.episode_time_s=500 \
  --dataset.reset_time_s=60 \
  --dataset.push_to_hub=true \
  --resume=true

