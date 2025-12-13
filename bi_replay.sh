#!/bin/bash

# Check if required arguments are provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <episode_id> [repo_id]"
    echo "Example: $0 0 tms-gvd/record-bi-test"
    echo "If repo_id is not provided, defaults to: tms-gvd/record-bi-test"
    exit 1
fi

EPISODE_ID=$1
REPO_ID=${2:-"tms-gvd/record-bi-test"}

lerobot-replay \
  --robot.type=bi_so101_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5AB90672281 \
  --robot.right_arm_port=/dev/tty.usbmodem59700731871 \
  --robot.left_arm_id=f1 \
  --robot.right_arm_id=f0 \
  --robot.id=bimanual_follower \
  --dataset.repo_id="$REPO_ID" \
  --dataset.episode="$EPISODE_ID"

