#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import time
from queue import Queue
from typing import Any
from pynput import keyboard

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

# from ..teleoperator import Teleoperator
# from ..utils import TeleopEvents
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig

from xarm.wrapper import XArmAPI

PYNPUT_AVAILABLE = True


class KeyboardTeleop():
    """
    Teleop class to use keyboard inputs for control.
    """
    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}

    @property
    def action_features(self) -> dict:
        return {
            "x": float,
            "y": float,
            "z": float,
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        pass

    def connect(self) -> None:
        if PYNPUT_AVAILABLE:
            print("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            print("pynput not available - skipping local keyboard listener.")
            self.listener = None

    def calibrate(self) -> None:
        pass

    def _on_press(self, key):
        if hasattr(key, "char"):
            self.event_queue.put((key.char, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            self.event_queue.put((key.char, False))
        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    def configure(self):
        pass
    
    # The action that the robotic arm will do
    def get_action(self) -> dict[str, Any]:
        before_read_t = time.perf_counter()

        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()

        # Generate action based on current key stateswwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
        action = list({key for key, val in self.current_pressed.items() if val})
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        x, y, z = 0, 0, 0 
        
        if action:
            key = action[0]

            if key=='w':
                z = 1
            elif key=='a':
                x = -1
            elif key=='s':
                z = -1
            elif key=='d':
                x = 1

        return {
            "x": x,
            "y": y,
            "z": z,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `robot.connect()` before `disconnect()`."
            )
        if self.listener is not None:
            self.listener.stop()
    


# if __name__ == "__main__":
#     config = KeyboardTeleopConfig()
#     my_keyboard = KeyboardTeleop(config)
    
#     # Connect to the keyboard listener
#     my_keyboard.connect()
    
#     # Check if it's connected (property, not method)
#     print(f"Connected: {my_keyboard.is_connected}")
    
#     try:
#         # Keep the program running and capture keyboard input
#         while True:
#             if my_keyboard.is_connected:
#                 action = my_keyboard.get_action()
#     except KeyboardInterrupt:
#         print("\nStopping...")
#     finally:
#         # Clean up
#         if my_keyboard.is_connected:
#             my_keyboard.disconnect()
#         print("Disconnected.")