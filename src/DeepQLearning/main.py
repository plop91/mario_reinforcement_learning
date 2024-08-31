import os
# import datetime
from logger import MetricLogger
from mario import Mario
from utils import *
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
# import gym_super_mario_bros
# from nes_py.wrappers import JoypadSpace
from pathlib import Path
# from gym.wrappers import FrameStack
# from gym.spaces import Box
# import gym
# import gymnasium as gym
# import random
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

TODO: Training regime
- Train on a single level until the model beats the level 50% of the time over the last 10 episodes, reset the exploration rate
- Train on all levels in world 1 until the model beats each level 50% of the time over the last 20 episodes, reset the exploration rate
- Train on all levels in world 1 and 2 until the model beats each level 50% of the time over the last 20 episodes, reset the exploration rate
- Train on all levels in the game(minus water levels) until the model beats each level 50% of the time over the last 20 episodes
"""


def main(args):
    # Create the environment
    env = setup_environment(args)

    # Check if we can use CUDA
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")

    if args.dir is not None:
        dir_name = args.dir
    else:
        if os.name == 'nt':
            dir_name = f"c:\\MarioTraining\\{args.name}"
        else:
            dir_name = f"~/MarioTraining/{args.name}"

    mario = Mario(state_dim=(4, 84, 84),
                      action_dim=env.action_space.n,
                      dir_name=dir_name,
                      distributed_training=False)
    

    if not args.no_gui:
        # create an opencv window to render the game
        cv2.startWindowThread()
        cv2.namedWindow("Super Mario Bros", cv2.WINDOW_NORMAL)

    # Create a logger
    logger = MetricLogger(dir_name)

    try:
        e = 0
        game_results = []
        print("Starting training...")
        while True:
            # Reset the environment
            state = env.reset()
            # number of steps in the episode, used to reduce the amount of frames rendered
            i = 0

            while True:
                # Run agent on the state
                action = mario.act(state)

                # Agent performs action
                next_state, reward, term, trunc, info = env.step(action)
                done = trunc or term

                if not args.no_gui:
                    # NOTE: the default render mode 'human' is faster with rendering times of 0.001s vs 0.01s for 'rgb_array'
                    # but the 'rgb_array' mode allows the game window to stay open after the game is done and does not rename the window
                    # which is required for streaming the training to twitch
                    if i % 2 == 0:
                        frame = env.render()
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        if not args.no_stats:
                            frame = write_stats_on_img(frame, args.name, e, i, reward)
                        cv2.imshow("Super Mario Bros", frame)
                        cv2.waitKey(1)

                # Remember
                mario.cache(state, next_state, action, reward, done)

                # Learn
                if args.mode == "train":
                    q, loss = mario.learn()

                # Logging
                logger.log_step(reward, loss, q)

                # Update state
                state = next_state

                # Check if end of game
                if done or info["flag_get"]:
                    if info["flag_get"]:
                        print("Level clear!")
                        game_results.append(info["flag_get"])
                    break
                i += 1
            
            # Log the episode
            logger.log_episode()
            
            if e % 20 == 0 and e > 0:
                logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
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
    parser.add_argument("regime", default=0, type=int, help="Regime to train on")
    parser.add_argument("--mode", default="train",
                        type=str, help="train | test")
    parser.add_argument("--dir", default=None, type=str,
                        help="Path to load/save the model")
    parser.add_argument("--no_gui", action='store_true', help="Do not display the game window")
    parser.add_argument("--no_stats", action='store_true', help="Do not display stats on the screen")
    # TODO: implement the limit_fps function
    parser.add_argument("--limit_fps", action='store_true', help="Limit the frames per second to 60")

    args = parser.parse_args()

    main(args)
