#!/usr/bin/env bash

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    set -a  # automatically export all variables
    source .env
    set +a  # stop automatically exporting
fi

# Set defaults if not provided in .env
FPS=${FPS:-30}
CAM_FRONT_INDEX=${CAM_FRONT_INDEX:-6}
CAM_TOP_INDEX=${CAM_TOP_INDEX:-4}
CAM_WIDTH=${CAM_WIDTH:-640}
CAM_HEIGHT=${CAM_HEIGHT:-480}
# CAM_FRONT="{type: opencv, index_or_path: ${CAM_FRONT_INDEX}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}}"
# CAM_TOP="{type: opencv, index_or_path: ${CAM_TOP_INDEX}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}}"
CAM_FRONT="{type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}"

# Check required variables
if [ -z "$PORT_FOLLOWER" ] || [ -z "$PORT_LEADER" ] || [ -z "$ID_FOLLOWER" ] || [ -z "$ID_LEADER" ]; then
    echo "Error: Required environment variables not set!"
    echo "Please set the following in your .env file:"
    echo "  PORT_FOLLOWER"
    echo "  PORT_LEADER"
    echo "  ID_FOLLOWER"
    echo "  ID_LEADER"
    exit 1
fi

lerobot-record \
    --robot.type=so101_follower \
    --robot.port="${PORT_FOLLOWER}" \
    --robot.id="${ID_FOLLOWER}" \
    --teleop.type=so101_leader \
    --teleop.port="${PORT_LEADER}" \
    --teleop.id="${ID_LEADER}" \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/test-camera \
    --dataset.num_episodes=5 \
    --dataset.single_task="Test" \
    --robot.cameras="{ front: ${CAM_FRONT} }" \
    --dataset.push_to_hub=true