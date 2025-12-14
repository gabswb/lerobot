#!/usr/bin/env python

"""
DAGger-style data collection with human interventions.

Flow per episode:
1) (Optional) Use teleoperation to set the initial state.
2) Press 'p' to start the policy rollout (this starts recording the episode).
3) During rollout, hold SPACE to take over with teleoperation; release SPACE to return to policy.
4) End the rollout when the time limit is reached or press 'q' to end early.
5) Decide whether to save or skip the episode.

Keys (global):
- SPACE: toggle between policy and teleop during rollout
- p: start policy rollout (from the pre-rollout teleop phase)
- q: end the current rollout early (go to save/skip prompt)
- Right arrow: end episode and save
- Left arrow: end episode and re-record (go back to pre-rollout teleop)
- ESC: stop the script after the current loop iteration
"""

import sys
import time
from pathlib import Path

#######################################################################################
# Configuration (edit these)
#######################################################################################

# Policy
POLICY_TYPE = "act"  # "act" or "smolvla"
POLICY_ID = "tms-gvd/act-scan-v2-150-20k"  # HF repo id or local path
TASK = ""  # Optional task string passed to the policy (often unused for ACT)

# Robot (bimanual SO101 follower)
LEFT_ARM_PORT = "/dev/f0"  # TODO: set
RIGHT_ARM_PORT = "/dev/f1"  # TODO: set
ROBOT_ID = "bimanual_follower"
LEFT_ARM_ID = "f0"  # optional calibration id, else uses f"{ROBOT_ID}_left"
RIGHT_ARM_ID = "f1"  # optional calibration id, else uses f"{ROBOT_ID}_right"
USE_DEGREES = False
LEFT_ARM_MAX_RELATIVE_TARGET = None  # e.g. 5.0 for safety clipping, or None
RIGHT_ARM_MAX_RELATIVE_TARGET = None  # e.g. 5.0 for safety clipping, or None

# Teleop (bimanual SO101 leader)
LEFT_TELEOP_PORT = "/dev/l0"  # TODO: set
RIGHT_TELEOP_PORT = "/dev/l1"  # TODO: set
TELEOP_ID = "bimanual_leader"
LEFT_TELEOP_ID = "l0"  # optional calibration id, else uses f"{TELEOP_ID}_left"
RIGHT_TELEOP_ID = "l1"  # optional calibration id, else uses f"{TELEOP_ID}_right"

# Rollout settings
FPS = 15
EPISODE_TIME_S = 60.0
NUM_EPISODES = 10

# Cameras (OpenCV). Set to [] to disable.
# Each entry: {"name": ..., "index": ..., "width": ..., "height": ..., "fps": ...} (fps can be None)
CAMERAS = [
    {"name": "left", "index": 4, "width": 640, "height": 480, "fps": 30},
    {"name": "right", "index": 2, "width": 640, "height": 480, "fps": 30},
    {"name": "top", "index": 8, "width": 640, "height": 480, "fps": 30},
    {"name": "scanner", "index": 6, "width": 640, "height": 480, "fps": 30},
]# Reset phase (teleop-only) duration between saved episodes.
RESET_TIME_S = 10.0

# Dataset recording (set DATASET_DIR=None to disable recording)
DATASET_REPO_ID = "tms-gvd/test-dagger"
DATASET_VIDEO = True
RECORD_PRE_ROLLOUT = False

# Visualization (rerun)
DISPLAY_DATA = False

# When the policy is controlling the follower, also command the leader arms to the same joint targets.
# This keeps leader and follower aligned so teleop takeover doesn't cause a jump.
SYNC_LEADER_DURING_POLICY = True


class KeyState:
    def __init__(self):
        self.intervene = False
        self.start_rollout = False
        self.end_episode = False
        self.save_episode = False
        self.rerecord_episode = False
        self.stop_all = False
        self.last_mode = "policy"  # for change detection only
        self._space_down = False


def _make_keyboard_listener(keys):
    try:
        from pynput import keyboard
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pynput is required for SPACE-bar intervention control. "
            "Install it or run in a non-headless environment."
        ) from e

    def on_press(key):
        if key == keyboard.Key.space:
            if not keys._space_down:
                keys.intervene = not keys.intervene
                keys._space_down = True
        elif key == keyboard.Key.right:
            keys.end_episode = True
            keys.save_episode = True
        elif key == keyboard.Key.left:
            keys.end_episode = True
            keys.rerecord_episode = True
        elif key == keyboard.Key.esc:
            keys.end_episode = True
            keys.save_episode = True
            keys.stop_all = True
        else:
            try:
                if key.char == "p":
                    keys.start_rollout = True
                elif key.char == "q":
                    keys.end_episode = True
            except Exception:
                return

    def on_release(key):
        if key == keyboard.Key.space:
            keys._space_down = False

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener


def _prompt_save_episode(default_save):
    prompt = "Save episode? [Y/n]: " if default_save else "Save episode? [y/N]: "
    while True:
        resp = input(prompt).strip().lower()
        if resp == "":
            return default_save
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")

def _sync_bimanual_leader_to_follower_action(bi_leader, follower_sent_action):
    left_goal = {}
    right_goal = {}
    for key, val in follower_sent_action.items():
        if not (isinstance(key, str) and key.endswith(".pos")):
            continue
        if key.startswith("left_"):
            motor = key.removeprefix("left_").removesuffix(".pos")
            left_goal[motor] = val
        elif key.startswith("right_"):
            motor = key.removeprefix("right_").removesuffix(".pos")
            right_goal[motor] = val

    if left_goal:
        bi_leader.left_arm.bus.sync_write("Goal_Position", left_goal)
    if right_goal:
        bi_leader.right_arm.bus.sync_write("Goal_Position", right_goal)

def _disable_leader_torque(bi_leader):
    """Disable torque on leader arms to allow free movement during teleop."""
    bi_leader.left_arm.bus.disable_torque()
    bi_leader.right_arm.bus.disable_torque()

def _enable_leader_torque(bi_leader):
    """Enable torque on leader arms to allow position control during policy mode."""
    bi_leader.left_arm.bus.enable_torque()
    bi_leader.right_arm.bus.enable_torque()

def _reset_phase(robot, teleop, robot_action_processor, robot_observation_processor, teleop_action_processor, keys, log_rerun_data, precise_sleep):
    if RESET_TIME_S is None or RESET_TIME_S <= 0:
        return

    print(f"Reset phase ({RESET_TIME_S}s). Teleoperate to reset, right arrow to finish early.")
    # Disable leader torque for free movement during reset phase
    _disable_leader_torque(teleop)
    
    start_t = time.perf_counter()
    while (time.perf_counter() - start_t) < RESET_TIME_S and not keys.stop_all:
        if keys.end_episode:
            keys.end_episode = False
            if keys.save_episode:
                keys.save_episode = False
            if keys.rerecord_episode:
                keys.rerecord_episode = False
            break

        t0 = time.perf_counter()
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        act_teleop = teleop.get_action()
        act_teleop_processed = teleop_action_processor((act_teleop, obs))
        act_to_send = robot_action_processor((act_teleop_processed, obs))
        robot.send_action(act_to_send)

        if DISPLAY_DATA:
            log_rerun_data(observation=obs_processed, action=act_to_send)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


def main():
    try:
        import lerobot  # noqa: F401
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parent
        sys.path.insert(0, str(repo_root / "src"))

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.policies.utils import make_robot_action
    from lerobot.processor import make_default_processors
    from lerobot.robots import make_robot_from_config
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.control_utils import predict_action
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import get_safe_torch_device
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

    from lerobot.robots.bi_so101_follower.config_bi_so101_follower import BiSO101FollowerConfig
    from lerobot.teleoperators.bi_so101_leader.config_bi_so101_leader import BiSO101LeaderConfig

    cameras = {
        cam["name"]: OpenCVCameraConfig(
            index_or_path=cam["index"],
            width=cam["width"],
            height=cam["height"],
            fps=FPS if cam.get("fps") is None else cam["fps"],
        )
        for cam in CAMERAS
    }

    if DISPLAY_DATA:
        init_rerun(session_name="dagger_collect")

    robot_cfg = BiSO101FollowerConfig(
        id=ROBOT_ID,
        left_arm_port=LEFT_ARM_PORT,
        right_arm_port=RIGHT_ARM_PORT,
        left_arm_id=LEFT_ARM_ID,
        right_arm_id=RIGHT_ARM_ID,
        cameras=cameras,
        left_arm_use_degrees=USE_DEGREES,
        right_arm_use_degrees=USE_DEGREES,
        left_arm_max_relative_target=LEFT_ARM_MAX_RELATIVE_TARGET,
        right_arm_max_relative_target=RIGHT_ARM_MAX_RELATIVE_TARGET,
    )
    teleop_cfg = BiSO101LeaderConfig(
        id=TELEOP_ID,
        left_arm_port=LEFT_TELEOP_PORT,
        right_arm_port=RIGHT_TELEOP_PORT,
        left_arm_id=LEFT_TELEOP_ID,
        right_arm_id=RIGHT_TELEOP_ID,
    )

    robot = make_robot_from_config(robot_cfg)
    teleop = make_teleoperator_from_config(teleop_cfg)

    print(f"Loading policy '{POLICY_TYPE}' from '{POLICY_ID}'...")
    policy_class = get_policy_class(POLICY_TYPE)
    policy = policy_class.from_pretrained(POLICY_ID)
    policy.eval()

    dataset = None
    if True:
        action_features = hw_to_dataset_features(robot.action_features, ACTION, use_video=DATASET_VIDEO)
        obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=DATASET_VIDEO)
        dataset_features = {**action_features, **obs_features}

        dataset = LeRobotDataset.create(
            repo_id=DATASET_REPO_ID,
            fps=FPS,
            features=dataset_features,
            root=None,
            robot_type=robot.name,
            use_videos=DATASET_VIDEO,
            image_writer_threads=(4 * len(cameras)) if cameras else 0,
        )
    else:
        # Minimal features dict for inference (no dataset recording).
        action_features = hw_to_dataset_features(robot.action_features, ACTION)
        obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
        dataset_features = {**action_features, **obs_features}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=POLICY_ID,
        dataset_stats=None,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    robot.connect()
    teleop.connect()
    if not robot.is_connected or not teleop.is_connected:
        raise RuntimeError("Robot and teleop must be connected before starting.")

    keys = KeyState()
    listener = _make_keyboard_listener(keys)

    device = get_safe_torch_device(policy.config.device)

    print(
        "\nControls:\n"
        "- Pre-rollout: teleoperate to set initial state, then press 'p' to start rollout\n"
        "- During rollout: press SPACE to toggle between teleop and policy\n"
        "- Press 'q' to end the rollout early (then choose save/skip)\n"
        "- Right arrow: end episode and save\n"
        "- Left arrow: end episode and re-record\n"
        "- ESC: end episode, save, and exit\n"
    )

    try:
        saved_episodes = 0
        attempt_idx = 0
        while saved_episodes < NUM_EPISODES:
            if keys.stop_all:
                break

            if dataset is not None:
                dataset.clear_episode_buffer()

            keys.start_rollout = False
            keys.end_episode = False
            keys.save_episode = False
            keys.rerecord_episode = False

            attempt_idx += 1
            print(f"\nEpisode {saved_episodes + 1}/{NUM_EPISODES}: pre-rollout teleop (press 'p' to start)")
            # Disable leader torque for free movement during pre-rollout teleop
            _disable_leader_torque(teleop)
            
            while not keys.start_rollout and not keys.stop_all:
                t0 = time.perf_counter()
                obs = robot.get_observation()
                obs_processed = robot_observation_processor(obs)
                act_teleop = teleop.get_action()
                act_teleop_processed = teleop_action_processor((act_teleop, obs))
                act_to_send = robot_action_processor((act_teleop_processed, obs))
                robot.send_action(act_to_send)

                if DISPLAY_DATA:
                    log_rerun_data(observation=obs_processed, action=act_to_send)

                if dataset is not None and RECORD_PRE_ROLLOUT:
                    obs_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
                    act_frame = build_dataset_frame(dataset.features, act_to_send, prefix=ACTION)
                    dataset.add_frame({**obs_frame, **act_frame, "task": TASK})

                precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

            if keys.stop_all:
                break

            keys.start_rollout = False
            keys.last_mode = "policy"
            keys.end_episode = False
            keys.save_episode = False
            keys.rerecord_episode = False

            print("Rollout started.")
            # Enable leader torque if syncing is enabled (starting in policy mode)
            if SYNC_LEADER_DURING_POLICY:
                _enable_leader_torque(teleop)
            
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

            start_t = time.perf_counter()
            intervened_steps = 0
            total_steps = 0

            while (time.perf_counter() - start_t) < EPISODE_TIME_S and not keys.end_episode and not keys.stop_all:
                t0 = time.perf_counter()
                obs = robot.get_observation()
                obs_processed = robot_observation_processor(obs)

                mode = "teleop" if keys.intervene else "policy"
                if mode != keys.last_mode:
                    if mode == "teleop":
                        print("Intervening (teleop).")
                        _disable_leader_torque(teleop)
                    else:
                        print("Back to policy.")
                        if SYNC_LEADER_DURING_POLICY:
                            _enable_leader_torque(teleop)
                    keys.last_mode = mode

                if mode == "teleop":
                    act = teleop.get_action()
                    act_processed = teleop_action_processor((act, obs))
                    intervened_steps += 1
                else:
                    observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
                    action_tensor = predict_action(
                        observation=observation_frame,
                        policy=policy,
                        device=device,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=policy.config.use_amp,
                        task=TASK if TASK else None,
                        robot_type=robot.robot_type,
                    )
                    act_processed = make_robot_action(action_tensor, dataset_features)

                act_to_send = robot_action_processor((act_processed, obs))
                sent_action = robot.send_action(act_to_send)
                if mode == "policy" and SYNC_LEADER_DURING_POLICY:
                    _sync_bimanual_leader_to_follower_action(teleop, sent_action)

                if DISPLAY_DATA:
                    log_rerun_data(observation=obs_processed, action=act_to_send)

                if dataset is not None:
                    obs_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
                    act_frame = build_dataset_frame(dataset.features, act_to_send, prefix=ACTION)
                    dataset.add_frame({**obs_frame, **act_frame, "task": TASK})

                total_steps += 1
                precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

            keys.end_episode = False
            if keys.stop_all:
                break

            default_save = intervened_steps > 0
            print(f"Episode finished. Steps={total_steps}, intervened_steps={intervened_steps}.")

            if dataset is None:
                continue

            if dataset.episode_buffer is None or dataset.episode_buffer.get("size", 0) == 0:
                print("No frames recorded; skipping save.")
                continue

            if keys.rerecord_episode:
                print("Re-record requested. Discarding episode and returning to pre-rollout teleop.")
                keys.rerecord_episode = False
                keys.save_episode = False
                dataset.clear_episode_buffer()
                continue

            if keys.save_episode:
                keys.save_episode = False
                dataset.save_episode()
                saved_episodes += 1
                print(f"Saved. Total episodes in dataset: {dataset.num_episodes}")
                if keys.stop_all:
                    break
                _reset_phase(
                    robot=robot,
                    teleop=teleop,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop_action_processor=teleop_action_processor,
                    keys=keys,
                    log_rerun_data=log_rerun_data,
                    precise_sleep=precise_sleep,
                )
                continue

            if _prompt_save_episode(default_save=default_save):
                dataset.save_episode()
                saved_episodes += 1
                print(f"Saved. Total episodes in dataset: {dataset.num_episodes}")
                _reset_phase(
                    robot=robot,
                    teleop=teleop,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop_action_processor=teleop_action_processor,
                    keys=keys,
                    log_rerun_data=log_rerun_data,
                    precise_sleep=precise_sleep,
                )
            else:
                dataset.clear_episode_buffer()
                print("Skipped.")
                _reset_phase(
                    robot=robot,
                    teleop=teleop,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop_action_processor=teleop_action_processor,
                    keys=keys,
                    log_rerun_data=log_rerun_data,
                    precise_sleep=precise_sleep,
                )

    finally:
        try:
            listener.stop()
        except Exception:
            pass

        try:
            teleop.disconnect()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
