#!/usr/bin/env python

"""
Simple script to run inference from a policy loaded from the Hugging Face Hub.
Edit the configuration variables below to set your policy, robot, and camera settings.

This script supports both synchronous and asynchronous inference modes.
- Synchronous: Traditional mode where robot waits for each action prediction
- Asynchronous: Decouples action prediction from execution, eliminating idle frames
"""

import argparse
import asyncio
import threading
import time
import torch
from typing import Any, Optional

# Configuration variables - edit these to match your setup
POLICY_TYPE = "act"  # Either "act" or "smolvla"
POLICY_ID = "tms-gvd/act-scan-v3-25k"  # Hugging Face Hub policy ID
LEFT_ARM_PORT = "/dev/f0"  # Left arm port /dev/tty.usbmodem59700731871 or /dev/ttyACM1
RIGHT_ARM_PORT = "/dev/f1"  # Right arm port /dev/tty.usbmodem5AB90672281 or /dev/ttyACM3
FPS = 30  # Frequency (Hz) for the rollout loop
LEFT_ARM_ID = "f0"  # Left arm ID
RIGHT_ARM_ID = "f1"  # Right arm ID

# Camera configuration - list of camera configs, or empty list to disable cameras
# Each camera config is a dict with: name, index, width, height, fps (fps can be None to use FPS)
CAMERAS = [
    {"name": "left", "index": 4, "width": 640, "height": 480, "fps": 30},
    {"name": "right", "index": 2, "width": 640, "height": 480, "fps": 30},
    {"name": "top", "index": 8, "width": 640, "height": 480, "fps": 30},
    {"name": "scanner", "index": 6, "width": 640, "height": 480, "fps": 30},
]
# Example with no cameras:
# CAMERAS = []

# Inference mode configuration
USE_ASYNC_INFERENCE = False  # Set to False to use synchronous inference

# Async inference configuration (only used if USE_ASYNC_INFERENCE=True)
SERVER_HOST = "127.0.0.1"  # Policy server host address
SERVER_PORT = 8080  # Policy server port
ACTIONS_PER_CHUNK = 100  # Number of actions per chunk
CHUNK_SIZE_THRESHOLD = 0.1  # Threshold for sending observations (0-1, lower = send more frequently)
# Note: chunk_size_threshold must be between 0 and 1. If you meant actions_per_chunk=20, that's set above.

# Duration limit (None = run indefinitely until Ctrl+C)
DURATION = None  # Duration in seconds, or None to run indefinitely

USE_TORCH_COMPILE = True 
TORCH_COMPILE_MODE = "default"

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.bi_so101_follower.config_bi_so101_follower import BiSO101FollowerConfig
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


class WebappController:
    """
    Control inference execution via WebSocket messages (from the Santa webapp).

    - action=start  => resume inference loop
    - action=pause  => pause inference loop (best-effort stop)
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._paused = threading.Event()
        self._paused.set()  # start paused in webapp mode
        self._last_pause_stop_ts = 0.0

    def is_paused(self) -> bool:
        return self.enabled and self._paused.is_set()

    def resume(self) -> None:
        if not self.enabled:
            return
        self._paused.clear()

    def pause(self) -> None:
        if not self.enabled:
            return
        self._paused.set()

    def should_send_stop(self, every_s: float = 0.4) -> bool:
        now = time.time()
        if now - self._last_pause_stop_ts >= every_s:
            self._last_pause_stop_ts = now
            return True
        return False


async def _ws_control_loop(ws_url: str, controller: WebappController) -> None:
    try:
        import websockets  # type: ignore
    except Exception as e:
        print(f"[webapp] Missing dependency 'websockets': {e}")
        return

    while True:
        try:
            async with websockets.connect(ws_url) as ws:
                print(f"[webapp] Connected to {ws_url}")
                # Tell dashboards our initial state.
                await ws.send(
                    '{"type":"robot_inference_status","source":"inference","state":"paused"}'
                )
                async for raw in ws:
                    try:
                        import json

                        msg = json.loads(raw)
                    except Exception:
                        continue
                    if msg.get("type") != "robot_inference_control":
                        continue
                    action = msg.get("action")
                    if not isinstance(action, str):
                        continue
                    action = action.strip().lower()
                    if action == "start":
                        controller.resume()
                        await ws.send(
                            '{"type":"robot_inference_status","source":"inference","state":"running"}'
                        )
                        print("[webapp] start")
                    elif action == "pause":
                        controller.pause()
                        await ws.send(
                            '{"type":"robot_inference_status","source":"inference","state":"paused"}'
                        )
                        print("[webapp] pause")
        except Exception as e:
            print(f"[webapp] Disconnected ({e}); retrying...")
            await asyncio.sleep(1.0)


def _start_ws_thread(ws_url: str, controller: WebappController) -> threading.Thread:
    t = threading.Thread(target=lambda: asyncio.run(_ws_control_loop(ws_url, controller)), daemon=True)
    t.start()
    return t


def _try_stop_robot(robot: Any, last_action: Optional[Any]) -> None:
    # Best-effort: prefer a native stop method if available.
    stop_fn = getattr(robot, "stop", None)
    if callable(stop_fn):
        try:
            stop_fn()
            return
        except Exception:
            pass

    if last_action is None:
        return

    zeroed = _zero_like(last_action)
    if zeroed is None:
        return
    try:
        robot.send_action(zeroed)
    except Exception:
        pass


def _zero_like(x: Any) -> Optional[Any]:
    # Recursively zero common action containers.
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None

    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            zv = _zero_like(v)
            if zv is None:
                return None
            out[k] = zv
        return out

    if np is not None and isinstance(x, np.ndarray):
        return np.zeros_like(x)

    if torch.is_tensor(x):
        return torch.zeros_like(x)

    if isinstance(x, (list, tuple)):
        seq = []
        for v in x:
            zv = _zero_like(v)
            if zv is None:
                return None
            seq.append(zv)
        return type(x)(seq)

    # Unknown type
    return None

if USE_ASYNC_INFERENCE:
    from lerobot.async_inference.configs import PolicyServerConfig, RobotClientConfig
    from lerobot.async_inference.policy_server import serve as serve_policy_server
    from lerobot.async_inference.robot_client import RobotClient
else:
    from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.utils import make_robot_action
    from lerobot.processor import make_default_processors
    from lerobot.robots import make_robot_from_config
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import get_safe_torch_device


def start_policy_server(host: str, port: int):
    """Start the policy server in a separate thread."""
    config = PolicyServerConfig(host=host, port=port, fps=FPS)
    serve_policy_server(config)


def run_async_inference(controller: WebappController):
    """Run inference using asynchronous mode."""
    print(f"Starting asynchronous inference with {POLICY_TYPE} policy: {POLICY_ID}")
    print(f"Policy server will run on {SERVER_HOST}:{SERVER_PORT}")
    
    # Initialize Rerun viewer
    init_rerun(session_name="async_inference")
    
    # Start policy server in a separate thread
    print("Starting policy server...")
    server_thread = threading.Thread(
        target=start_policy_server,
        args=(SERVER_HOST, SERVER_PORT),
        daemon=True
    )
    server_thread.start()
    
    # Give the server a moment to start
    time.sleep(2)
    print("Policy server started")

    print(f"Initializing bimanual SO101 robot on ports {LEFT_ARM_PORT} (left) and {RIGHT_ARM_PORT} (right)")
    # Configure cameras
    cameras_dict = {}
    for camera_config in CAMERAS:
        camera_fps = camera_config.get("fps")
        if camera_fps is None:
            camera_fps = FPS
        cameras_dict[camera_config["name"]] = OpenCVCameraConfig(
            index_or_path=camera_config["index"],
            width=camera_config["width"],
            height=camera_config["height"],
            fps=camera_fps,
        )
        print(f"Configured camera '{camera_config['name']}' with index {camera_config['index']} ({camera_config['width']}x{camera_config['height']} @ {camera_fps} fps)")
    
    # Create robot configuration
    robot_config = BiSO101FollowerConfig(
        left_arm_port=LEFT_ARM_PORT,
        right_arm_port=RIGHT_ARM_PORT,
        left_arm_id=LEFT_ARM_ID,
        right_arm_id=RIGHT_ARM_ID,
        cameras=cameras_dict
    )

    # Create robot client configuration
    client_config = RobotClientConfig(
        policy_type=POLICY_TYPE,
        pretrained_name_or_path=POLICY_ID,
        robot=robot_config,
        actions_per_chunk=ACTIONS_PER_CHUNK,
        chunk_size_threshold=CHUNK_SIZE_THRESHOLD,
        fps=FPS,
        server_address=f"{SERVER_HOST}:{SERVER_PORT}",
        policy_device="cpu",  # Change to "cuda" or "mps" if you have GPU/Apple Silicon
        task="",  # Task instruction (not used for ACT policies)
        aggregate_fn_name="weighted_average",
        debug_visualize_queue_size=False,
    )

    print(f"Starting robot client with actions_per_chunk={ACTIONS_PER_CHUNK}, chunk_size_threshold={CHUNK_SIZE_THRESHOLD}")
    print("Press Ctrl+C to stop")

    # Create robot client
    client = RobotClient(client_config)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()

        last_action = None
        try:
            # Custom control loop with Rerun logging
            client.start_barrier.wait()
            client.logger.info("Control loop thread starting")

            while client.running:
                control_loop_start = time.perf_counter()

                if controller.is_paused():
                    if controller.should_send_stop():
                        _try_stop_robot(client.robot, last_action)
                    time.sleep(max(0, client.config.environment_dt - (time.perf_counter() - control_loop_start)))
                    continue
                
                # Perform actions when available
                if client.actions_available():
                    performed_action = client.control_loop_action(verbose=False)
                    # Log action to Rerun
                    if performed_action:
                        last_action = performed_action
                        log_rerun_data(action=performed_action)

                # Stream observations to the remote policy server
                if client._ready_to_send_observation():
                    captured_observation = client.control_loop_observation(task=client_config.task, verbose=False)
                    # Log observation to Rerun
                    if captured_observation:
                        log_rerun_data(observation=captured_observation)

                # Dynamically adjust sleep time to maintain the desired control frequency
                time.sleep(max(0, client.config.environment_dt - (time.perf_counter() - control_loop_start)))

        except KeyboardInterrupt:
            print("\nStopping inference...")
        finally:
            client.stop()
            action_receiver_thread.join()
            if client_config.debug_visualize_queue_size:
                from lerobot.async_inference.helpers import visualize_action_queue_size
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")
            print("Done")


def run_sync_inference(controller: WebappController):
    """Run inference using synchronous mode."""
    print(f"Loading {POLICY_TYPE} policy from hub: {POLICY_ID}")
    
    # Initialize Rerun viewer
    init_rerun(session_name="sync_inference")
    
    # Get policy class and load from pretrained
    policy_class = get_policy_class(POLICY_TYPE)
    policy = policy_class.from_pretrained(POLICY_ID)
    policy.eval()
    
    if USE_TORCH_COMPILE:
        print(f"Compiling policy with torch.compile (mode={TORCH_COMPILE_MODE})...")
        try:
            policy = torch.compile(policy, mode=TORCH_COMPILE_MODE)
            print("Policy compiled successfully")
        except Exception as e:
            print(f"Warning: torch.compile failed ({e}), continuing without compilation")

    print(f"Initializing bimanual SO101 robot on ports {LEFT_ARM_PORT} (left) and {RIGHT_ARM_PORT} (right)")
    # Configure cameras
    cameras_dict = {}
    for camera_config in CAMERAS:
        camera_fps = camera_config.get("fps")
        if camera_fps is None:
            camera_fps = FPS
        cameras_dict[camera_config["name"]] = OpenCVCameraConfig(
            index_or_path=camera_config["index"],
            width=camera_config["width"],
            height=camera_config["height"],
            fps=camera_fps,
        )
        print(f"Configured camera '{camera_config['name']}' with index {camera_config['index']} ({camera_config['width']}x{camera_config['height']} @ {camera_fps} fps)")
    
    # Create robot configuration
    robot_config = BiSO101FollowerConfig(
        left_arm_port=LEFT_ARM_PORT,
        right_arm_port=RIGHT_ARM_PORT,
        left_arm_id=LEFT_ARM_ID,
        right_arm_id=RIGHT_ARM_ID,
        cameras=cameras_dict
    )

    robot = make_robot_from_config(robot_config)
    robot.connect()

    if not robot.is_connected:
        raise ValueError("Failed to connect to robot!")

    print("Robot connected successfully")

    # Create dataset features from robot
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    # Build preprocessor and postprocessor
    # The processors will try to load stats from the pretrained path if available
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=POLICY_ID,
        dataset_stats=None,  # Will be loaded from pretrained path if available
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    # Get default processors for robot
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    device = get_safe_torch_device(policy.config.device)
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    print(f"Starting synchronous inference loop at {FPS} Hz")
    print("Press Ctrl+C to stop")

    start_time = time.perf_counter()
    last_action_to_send = None
    try:
        while True:
            loop_start = time.perf_counter()

            # Check duration limit
            if DURATION is not None:
                elapsed = time.perf_counter() - start_time
                if elapsed >= DURATION:
                    print(f"Reached duration limit of {DURATION} seconds")
                    break

            if controller.is_paused():
                if controller.should_send_stop():
                    _try_stop_robot(robot, last_action_to_send)
                precise_sleep(1.0 / FPS - (time.perf_counter() - loop_start))
                continue

            # Get observation from robot
            obs = robot.get_observation()

            # Process observation
            obs_processed = robot_observation_processor(obs)

            # Log observation to Rerun
            log_rerun_data(observation=obs_processed)

            # Build observation frame for policy
            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

            # Predict action using policy
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                robot_type=robot.robot_type,
            )

            # Convert action to robot format
            robot_action = make_robot_action(action_values, dataset_features)

            # Process action for robot
            robot_action_to_send = robot_action_processor((robot_action, obs_processed))

            # Log action to Rerun
            log_rerun_data(action=robot_action_to_send)

            # Send action to robot
            robot.send_action(robot_action_to_send)
            last_action_to_send = robot_action_to_send

            # Maintain frequency
            dt = time.perf_counter() - loop_start
            precise_sleep(1.0 / FPS - dt)

    except KeyboardInterrupt:
        print("\nStopping inference...")
    finally:
        print("Disconnecting robot...")
        robot.disconnect()
        print("Done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--webapp",
        action="store_true",
        help="Enable WebSocket control from the Santa webapp (start/pause).",
    )
    parser.add_argument(
        "--ws-url",
        default="ws://127.0.0.1:8765",
        help="WebSocket URL for RobotVisionBeacon host server.",
    )
    args = parser.parse_args()

    controller = WebappController(enabled=bool(args.webapp))
    if args.webapp:
        _start_ws_thread(args.ws_url, controller)
        print("[webapp] Waiting for START command from webapp…")

    if USE_ASYNC_INFERENCE:
        run_async_inference(controller)
    else:
        run_sync_inference(controller)


if __name__ == "__main__":
    main()
