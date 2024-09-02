import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# import gymnasium as gym
# from gymnasium.spaces import Box
# from gymnasium.wrappers import FrameStack

import numpy as np

import torch
from torchvision import transforms as T

from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
# JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

import cv2

import os


def write_on_img(img, text, pos, font_scale=1, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), line_type=cv2.LINE_4):
    cv2.putText(img, text, (pos[0]+1, pos[1]+1), font, font_scale,
                (0, 0, 0), thickness, line_type)
    cv2.putText(img, text, pos, font, font_scale,
                color, thickness, line_type)
    return img


def write_stats_on_img(frame, name, e, i, reward):

    frame = write_on_img(frame,
                         f"Model's Name:{name}",
                         pos=(10, 50),
                         font=cv2.FONT_HERSHEY_TRIPLEX,
                         font_scale=.45,
                         color=(0, 150, 255),
                         thickness=1,
                         line_type=cv2.LINE_4)
    frame = write_on_img(frame,
                         f"Episode:{e} Step:{i}",
                         pos=(10, 70),
                         font=cv2.FONT_HERSHEY_TRIPLEX,
                         font_scale=.45,
                         color=(0, 150, 255),
                         thickness=1,
                         line_type=cv2.LINE_4)
    frame = write_on_img(frame,
                         f"Reward:{reward}",
                         pos=(10, 90),
                         font=cv2.FONT_HERSHEY_TRIPLEX,
                         font_scale=.45,
                         color=(0, 150, 255),
                         thickness=1,
                         line_type=cv2.LINE_4)

    return frame


def write_neat_info_on_img(frame, generation, genome, fitness):
    frame = write_on_img(frame,
                         f"Generation:{generation}",
                         pos=(10, 50),
                         font=cv2.FONT_HERSHEY_TRIPLEX,
                         font_scale=.45,
                         color=(0, 150, 255),
                         thickness=1,
                         line_type=cv2.LINE_4)
    frame = write_on_img(frame,
                         f"Genome:{genome}",
                         pos=(10, 70),
                         font=cv2.FONT_HERSHEY_TRIPLEX,
                         font_scale=.45,
                         color=(0, 150, 255),
                         thickness=1,
                         line_type=cv2.LINE_4)
    frame = write_on_img(frame,
                         f"Fitness:{fitness}",
                         pos=(10, 90),
                         font=cv2.FONT_HERSHEY_TRIPLEX,
                         font_scale=.45,
                         color=(0, 150, 255),
                         thickness=1,
                         line_type=cv2.LINE_4)

    return frame


def setup_environment(args):

    regimes = [['1-1'], ['1-1', '1-2', '1-3', '1-4'], ['1-1', '1-2', '1-3', '1-4', '2-1', '2-3', '2-4'],
               ['1-1', '1-2', '1-3', '1-4', '2-1', '2-3', '2-4', '3-1', '3-2', '3-3', '3-4', '4-1', '4-2', '4-3',
                '4-4', '5-1', '5-2', '5-3', '5-4', '6-1', '6-2', '6-3', '6-4', '7-1', '7-3', '7-4', '8-1', '8-2', '8-3', '8-4']]  # all stages minus the water levels

    # env = gym_super_mario_bros.make(f"SuperMarioBros-{args.world}-v0", render_mode='human', apply_api_compatibility=True)
    # env = gym_super_mario_bros.make(f"SuperMarioBros-{args.world}-v0", render_mode='rgb_array', apply_api_compatibility=True)
    # env = gym_super_mario_bros.make(f"SuperMarioBrosRandomStages-v0", stages=['1-1', '1-2', '1-3', '1-4'], render_mode='rgb_array', apply_api_compatibility=True)
    env = gym_super_mario_bros.make(f"SuperMarioBrosRandomStages-v0",
                                    stages=regimes[args.regime], render_mode='rgb_array', apply_api_compatibility=True)
    # env = gym_super_mario_bros.make(f"SuperMarioBrosRandomStages-v0", stages=stages)

    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env.reset()
    # next_state, reward, done, trunc, info = env.step(action=0)
    # print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
    return env


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        obs, total_reward, done, trunk, info = None, 0.0, False, None, None
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation
