import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.motors_bus import (
    MotorsBus
)
import numpy as np

from lerobot.robots import Robot
# from ..utils import ensure_safe_goal_position

# Custom
from lerobot.robots.pearlywhite.config_pearlywhite_follower import PearlyWhiteFollowerConfig
from xarm.wrapper import XArmAPI

from PIL import Image

import cv2

logger = logging.getLogger(__name__)


class PearlyWhiteFollower(Robot):
    config_class = PearlyWhiteFollowerConfig
    name = "pearlywhite_robot"
    id = 'pearlywhite'

    def __init__(self, config: PearlyWhiteFollowerConfig):
        self.config = config

        self.arm = XArmAPI(self.config.port, baud_checkset=False)

        self.cameras = make_cameras_from_configs(config.cameras)
    
    # These keys match the output of get_observation. xyz is the current position of the robotic arm
    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            "x": float,
            "y": float,
            "z": float
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }
    
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.arm.connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """

        self.arm.connect()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)   # position control mode
        self.arm.set_state(0)  # ready state

        # Go to initial position
        self.arm.set_position(x=9.97717, y=207.91037, z=190.492111, roll=180, pitch=0, yaw=90, wait=True, radius=-1, speed=200)

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.arm.connected:
            raise ConnectionError(f"{self} is not connected.")
        
        # Read joint angles in degrees
        current_pos = self.arm.get_position_aa()[1]
        obs_dict = {
            "x": current_pos[0],
            "y": current_pos[1],
            "z": current_pos[2]
        }


        # Capture images from cameras
        for cam_name, cam in self.cameras.items():
            frame = cam.async_read()
            obs_dict[cam_name] = frame

            # cv2.imwrite("image.png", obs_dict[cam_key])

        return obs_dict


    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.arm.connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not action == {}:
            x, y, z = action["x"], action["y"], action["z"]

            current_pos = self.arm.get_position()[1]

            self.arm.set_position(x=current_pos[0]+x, y=current_pos[1]+y, z=current_pos[2]+z, roll=180, pitch=0, yaw=90, wait=True, radius=-1, speed=200)
        return action
    

    def disconnect(self):
        self.arm.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")


# if __name__ == "__main__":

#     config = PearlyWhiteFollowerConfig()
#     robot = PearlyWhiteFollower(config)   # create an instance
#     # joint_positions = {
#     #     "x": -15,
#     #     "y": 0,
#     #     "z": 0
#     # }
#     # robot.connect()
#     # robot._cameras_ft
#     # robot.get_observation()
#     # robot.send_action(joint_positions)     # call its method
#     robot.disconnect()