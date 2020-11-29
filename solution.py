#!/usr/bin/env python3

import io
import os

import numpy as np
from PIL import Image

from aido_schemas import (Context, DB20Commands, DB20Observations, EpisodeStart, JPGImage, LEDSCommands,
                          protocol_agent_DB20, PWMCommands, RGB, wrap_direct)
from model import TfInference

expect_shape = (480, 640, 3)


class TensorflowTemplateAgent:
    current_image: np.ndarray
    model: TfInference

    def __init__(self, load_model=False, model_path=None):
        pass

    def init(self, context: Context):
        context.info('Check GPU avaialbility...')
        self.check_tensorflow_gpu(context)
        context.info('init()')
        # define observation and output shapes
        self.model = TfInference(observation_shape=(1,) + expect_shape,
                                 # this is the shape of the image we get.
                                 action_shape=(1, 2),  # we need to output v, omega.
                                 graph_location='tf_models/')  # this is the folder where our models are
        # stored.
        self.current_image = np.zeros(expect_shape)

    def check_tensorflow_gpu(self, context: Context):
        req = os.environ.get('AIDO_REQUIRE_GPU', None)

        import tensorflow as tf
        name = tf.test.gpu_device_name()
        context.info(f'gpu_device_name: {name!r} AIDO_REQUIRE_GPU = {req!r}')
        if req is not None:
            if not name:  # None or ''
                msg = 'Could not find gpu device.'
                context.error(msg)
                raise RuntimeError(msg)

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
        context.write('commands', commands)

    def finish(self, context: Context):
        context.info('finish()')


def jpg2rgb(image_data: bytes) -> np.ndarray:
    """ Reads JPG bytes as RGB"""

    im = Image.open(io.BytesIO(image_data))
    im = im.convert('RGB')
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data


def main():
    node = TensorflowTemplateAgent()
    protocol = protocol_agent_DB20
    wrap_direct(node=node, protocol=protocol)


if __name__ == '__main__':
    main()
