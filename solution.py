#!/usr/bin/env python3

import io
import os
from typing import Tuple

import numpy as np
from PIL import Image

from aido_schemas import (
    Context,
    DB20Commands,
    DB20Observations,
    EpisodeStart,
    JPGImage,
    LEDSCommands,
    logger,
    no_hardware_GPU_available,
    protocol_agent_DB20,
    PWMCommands,
    RGB,
    wrap_direct,
)


class TensorflowTemplateAgent:
    current_image: np.ndarray

    def __init__(self, expect_shape: Tuple[int, int, int] = (480, 640, 3)):
        self.expect_shape = expect_shape

    def init(self, context: Context):
        context.info("Checking GPU availability...")
        limit_gpu_memory()
        self.check_tensorflow_gpu(context)

        from model import TfInference

        # this is the folder where our models are
        graph_location = "tf_models/"
        # define observation and output shapes
        self.model = TfInference(
            observation_shape=(1,) + self.expect_shape,
            # this is the shape of the image we get.
            action_shape=(1, 2),  # we need to output v, omega.
            graph_location=graph_location,
        )
        # stored.
        self.current_image = np.zeros(self.expect_shape)

    def check_tensorflow_gpu(self, context: Context):

        import tensorflow as tf

        name = tf.test.gpu_device_name()
        context.info(f"gpu_device_name: {name!r} ")
        if not name:  # None or ''
            no_hardware_GPU_available(context)

    def on_received_seed(self, data: int):
        np.random.seed(data)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        context.info(f'Starting episode "{data.episode_name}".')

    def on_received_observations(self, data: DB20Observations):
        camera: JPGImage = data.camera
        self.current_image = jpg2rgb(camera.jpg_data)

    def compute_action(self, observation):
        action = self.model.predict(observation)
        return action.astype(float)

    def on_received_get_commands(self, context: Context):
        pwm_left, pwm_right = self.compute_action(self.current_image)

        pwm_left = float(np.clip(pwm_left, -1, +1))
        pwm_right = float(np.clip(pwm_right, -1, +1))

        grey = RGB(0.0, 0.0, 0.0)
        led_commands = LEDSCommands(grey, grey, grey, grey, grey)
        pwm_commands = PWMCommands(motor_left=pwm_left, motor_right=pwm_right)
        commands = DB20Commands(pwm_commands, led_commands)
        context.write("commands", commands)

    def finish(self, context: Context):
        context.info("finish()")


def jpg2rgb(image_data: bytes) -> np.ndarray:
    """Reads JPG bytes as RGB"""

    im = Image.open(io.BytesIO(image_data))
    im = im.convert("RGB")
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data


def limit_gpu_memory(memory_limit: int = 1024):
    """Restricts TensorFlow to only allocated 1GB of memory on the first GPU"""
    import tensorflow as tf

    physical_gpus = tf.config.experimental.list_physical_devices("GPU")
    if physical_gpus:
        try:
            c = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
            tf.config.experimental.set_virtual_device_configuration(physical_gpus[0], c)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            logger.info(num_physical_gpus=len(physical_gpus), num_logical_gpus=len(logical_gpus))
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            logger.error(e)


def main():
    node = TensorflowTemplateAgent()
    protocol = protocol_agent_DB20
    wrap_direct(node=node, protocol=protocol)


if __name__ == "__main__":
    main()
