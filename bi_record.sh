#!/bin/bash

lerobot-record \
  --robot.type=bi_so101_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5AB90672281 \
  --robot.right_arm_port=/dev/tty.usbmodem59700731871 \
  --robot.left_arm_id=f1 \
  --robot.right_arm_id=f0 \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}
  }' \
  --teleop.type=bi_so101_leader \
  --teleop.left_arm_port=/dev/tty.usbmodem5A460819611 \
  --teleop.right_arm_port=/dev/tty.usbmodem5AB90680581 \
  --teleop.left_arm_id=l0 \
  --teleop.right_arm_id=l1 \
  --display_data=true \
  --dataset.repo_id="tms-gvd/record-bi-test" \
  --dataset.num_episodes=25 \
  --dataset.single_task="Bimanual manipulation task" \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=60 \
  --dataset.push_to_hub=true

