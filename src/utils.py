import gym
from gym.spaces import Box
import numpy as np
import torch
from torchvision import transforms as T
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack
import os


def setup_environment(args):
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make(
            f"SuperMarioBros-{args.world}-v0", new_step_api=True)
    else:
        stages = ['1-1', '1-2', '1-3', '1-4', '2-1', '2-2', '2-3', '2-4', '3-1', '3-2', '3-3', '3-4', '4-1', '4-2', '4-3',
                    '4-4', '5-1', '5-2', '5-3', '5-4', '6-1', '6-2', '6-3', '6-4', '7-1', '7-2', '7-3', '7-4', '8-1', '8-2', '8-3', '8-4']
        if args.no_gui:
            # env = gym_super_mario_bros.make(
            #     f"SuperMarioBros-{args.world}-v0", apply_api_compatibility=True)
            env = gym_super_mario_bros.make(
                f"SuperMarioBrosRandomStages-v0", stages=stages, apply_api_compatibility=True)
            
        else:
            # env = gym_super_mario_bros.make(f"SuperMarioBros-{args.world}-v0", render_mode='human', apply_api_compatibility=True)
            # env = gym_super_mario_bros.make(f"SuperMarioBros-{args.world}-v0", render_mode='rgb_array', apply_api_compatibility=True)
            # env = gym_super_mario_bros.make(f"SuperMarioBrosRandomStages-v0", stages=['1-1', '1-2', '1-3', '1-4'], render_mode='rgb_array', apply_api_compatibility=True)
            env = gym_super_mario_bros.make(
                f"SuperMarioBrosRandomStages-v0", stages=stages, render_mode='rgb_array', apply_api_compatibility=True)

    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env.reset()
    # next_state, reward, done, trunc, info = env.step(action=0)
    # print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)

    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)
    return env


def setup_save_load_dirs(args):
    if args.save_dir is None and args.load_dir is None:
        if os.name == 'nt':
            load_dir = f"c:\\MarioTraining\\{args.name}"
            save_dir = f"c:\\MarioTraining\\{args.name}"
        else:
            load_dir = f"/tmp/MarioTraining/{args.name}"
            save_dir = f"/tmp/MarioTraining/{args.name}"

    elif args.save_dir is not None and args.load_dir is not None:
        save_dir = args.save_dir
        load_dir = args.load_dir

    elif args.save_dir is not None:
        save_dir = args.save_dir
        if os.name == 'nt':
            load_dir = f"c:\\MarioTraining\\{args.name}"
        else:
            load_dir = f"/tmp/MarioTraining/{args.name}"

    elif args.load_dir is not None:
        load_dir = args.load_dir
        if os.name == 'nt':
            save_dir = f"c:\\MarioTraining\\{args.name}"
        else:
            save_dir = f"/tmp/MarioTraining/{args.name}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(load_dir):
        os.makedirs(load_dir)

    return save_dir, load_dir


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
