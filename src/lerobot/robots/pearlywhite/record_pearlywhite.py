from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features

from lerobot.robots.pearlywhite.pearlywhite_follower import PearlyWhiteFollower
from lerobot.robots.pearlywhite.config_pearlywhite_follower import PearlyWhiteFollowerConfig
from lerobot.teleoperators.pearlywhite_keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig

from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun as _init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import (
    make_default_processors,
)
from huggingface_hub import login
from dotenv import load_dotenv
import os

NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "Pick and Place Object Detection"

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

# Create the robot and teleoperator configurations
camera_config = {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS)}
robot_config = PearlyWhiteFollowerConfig()
teleop_config = KeyboardTeleopConfig()

# Initialize the robot and teleoperator
robot = PearlyWhiteFollower(robot_config)
teleop = KeyboardTeleop(teleop_config)

teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action", use_video=False)
obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=False)
dataset_features = {**action_features, **obs_features}

my_repo = "rdteteam/hello3"

# Create the dataset
dataset = LeRobotDataset.create(
    repo_id=my_repo,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=False,
    image_writer_threads=4,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
_init_rerun(session_name="recording")

# Connect the robot and teleoperator
robot.connect()
teleop.connect()

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
    
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording")
robot.disconnect()
teleop.disconnect()
dataset.push_to_hub(repo_id=my_repo, private=True)