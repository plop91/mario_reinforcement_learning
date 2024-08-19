import os
import datetime
from logger import MetricLogger
from mario import Mario
from utils import *
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from pathlib import Path
from gym.wrappers import FrameStack
from gym.spaces import Box
import gym
import random
import warnings
import cv2
import time
warnings.filterwarnings("ignore")


"""
TODO:
- Name the model
- Save the model 
- Load the model
- Save the weights of the model
- Load the weights of the model
- Save the model after every x episodes based on argument
- Load the model from a specific file based on argument
- when a model beats the game, save it
- when a model beats the game multiple times in a row, save it and move on to the next level
- add 'training' file to the save directory that contains information about the training

"""


def main(args):
    # Create the environment
    env = setup_environment(args)

    # Check if we can use CUDA
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")

    # setup the save and load directories
    save_dir, load_dir = setup_save_load_dirs(args)

    # TODO: remove the requirement a cast to path object here, remnant of the old code
    save_dir = Path(save_dir)
    print(f"Save directory: {save_dir}   Load directory: {load_dir}")

    files = os.listdir(load_dir)
    load_model = False
    checkpoint = None
    if len(files) > 0:
        # if there is a model in the save directory, load it
        for file in files:
            if file.endswith(".model"):
                load_model = True
                break
    
        # if there is a checkpoint in the load directory, load it
        max_checkpoint = 0
        for file in files:
            if file.endswith(".chkpt"):
                i = int(file.replace("mario_net_", "").replace(".chkpt", ""))
                if i >= max_checkpoint:
                    max_checkpoint = i
                    checkpoint = os.path.join(load_dir, file)

    mario = Mario(state_dim=(4, 84, 84),
                      action_dim=env.action_space.n,
                      save_dir=save_dir,
                      load_model=load_model)
    
    if checkpoint is not None:
        mario.load(checkpoint)

    if not args.no_gui:
        # create an opencv window to render the game
        cv2.startWindowThread()
        cv2.namedWindow("Super Mario Bros", cv2.WINDOW_NORMAL)

    try:
        e = 0
        game_results = []
        while True:
            # Reset the environment
            state = env.reset()
            # number of steps in the episode, used to reduce the amount of frames rendered
            i = 0
            # TODO: remove the times list, it is only used for debugging
            times = []

            while True:
                # Run agent on the state
                action = mario.act(state)

                # Agent performs action
                next_state, reward, done, trunc, info = env.step(action)

                if not args.no_gui:
                    # NOTE: the default render mode 'human' is faster with rendering times of 0.001s vs 0.01s for 'rgb_array'
                    # but the 'rgb_array' mode allows the game window to stay open after the game is done and does not rename the window
                    # which is required for streaming the training to twitch
                    # TODO: remove all the timing code, it is only used for debugging
                    start_time = time.time()
                    # env.render()
                    if i % 2 == 0:
                        frame = env.render()
                        cv2.imshow("Super Mario Bros", cv2.cvtColor(
                            frame, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                    end_time = time.time()
                    total_time = end_time - start_time
                    times.append(total_time)
                    # if i % 100 == 0:
                    #     print(f"Average render time: {sum(times) / len(times)}")

                # Remember
                mario.cache(state, next_state, action, reward, done)

                # Learn
                if args.mode == "train":
                    q, loss = mario.learn()

                # Logging
                # TODO: enable the logger
                # logger.log_step(reward, loss, q)

                # Update state
                state = next_state

                # Check if end of game
                if done or info["flag_get"]:
                    if info["flag_get"]:
                        print("Level clear!")
                        game_results.append(info["flag_get"])
                    break
                i += 1

            # logger.log_episode()
            # if (e % 20 == 0) or (e == episodes - 1):
            if e % 20 == 0:
                # logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
                print(
                    f"Episode: {e}, Step: {mario.curr_step}, Exploration rate: {mario.exploration_rate}")
            e += 1

    except KeyboardInterrupt:
        print("Keyboard interrupt")
        print("Saving model...")
        mario.save()
        print("Model saved")

    finally:
        env.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("name", help="Name of the model")
    parser.add_argument("--stage", default="1-1", type=str,
                        help="World to train on Default: 1-1")
    parser.add_argument("-r", "--randomize_stage",
                        action='store_true', help="Randomize the stage")
    parser.add_argument("--mode", default="train",
                        type=str, help="train | test")
    parser.add_argument("--load_dir", default=None, type=str,
                        help="Path to load a model from if not in the default location")
    parser.add_argument("--save_dir", default=None, type=str,
                        help="Path to save the model if not in the default location")
    parser.add_argument("--no_gui", action='store_true')

    args = parser.parse_args()

    main(args)
