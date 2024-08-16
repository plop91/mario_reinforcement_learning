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
import warnings
warnings.filterwarnings("ignore")


# NES Emulator for OpenAI Gym

# Super Mario environment for OpenAI Gym


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
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make(
            "SuperMarioBros-1-1-v0", new_step_api=True)
    else:
        if args.no_gui:
            env = gym_super_mario_bros.make(
                "SuperMarioBros-1-1-v0", apply_api_compatibility=True)
        else:
            env = gym_super_mario_bros.make(
                "SuperMarioBros-1-1-v0", render_mode='human', apply_api_compatibility=True)

    
    env = JoypadSpace(env, COMPLEX_MOVEMENT)

    env.reset()
    next_state, reward, done, trunc, info = env.step(action=0)
    # print(f"{next_state.shape},\n {reward},\n {done},\n {info}")

    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)

    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)

    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    if args.save_dir is None and args.load_dir is None:
        if os.name == 'nt':
            load_dir = f"c:\MarioTraining\{args.name}"
            save_dir = f"c:\MarioTraining\{args.name}"
        else:
            load_dir = f"/tmp/MarioTraining/{args.name}"
            save_dir = f"/tmp/MarioTraining/{args.name}"

    elif args.save_dir is not None and args.load_dir is not None:
        save_dir = args.save_dir
        load_dir = args.load_dir
    
    elif args.save_dir is not None:
        save_dir = args.save_dir
        if os.name == 'nt':
            load_dir = f"c:\MarioTraining\{args.name}"
        else:
            load_dir = f"/tmp/MarioTraining/{args.name}"

    elif args.load_dir is not None:
        load_dir = args.load_dir
        if os.name == 'nt':
            save_dir = f"c:\MarioTraining\{args.name}"
        else:
            save_dir = f"/tmp/MarioTraining/{args.name}"
    save_dir = Path(save_dir)
    print(f"Save directory: {save_dir}   Load directory: {load_dir}")
    # check if we are loading a model
    if os.path.exists(load_dir):
        # we are loading a model
        print(f"Loading model from {load_dir}")
        # load the checkpoint
        mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, load_model=True)
        # get most recent checkpoint
        files = os.listdir(load_dir)
        checkpoint = None
        max_checkpoint = 0
        for file in files:
            if file.endswith(".chkpt"):
                i = int(file.replace("mario_net_", "").replace(".chkpt", ""))
                if i >= max_checkpoint:
                    max_checkpoint = i
                    checkpoint = os.path.join(load_dir, file)
        if checkpoint is None:
            raise ValueError(f"No checkpoint found in {load_dir}")
        
        mario.load(checkpoint)

    else:
        # we are not loading a model
        print(f"Creating new model: {args.name}")
        save_dir.mkdir(parents=True)
        mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

        # print the model summary
        print(mario.net)

    # mario = Mario(state_dim=(4, 84, 84),
    #               action_dim=env.action_space.n, save_dir=save_dir)

    # # if args.load_dir is not None:
    # if True:
    #     # list all folders in models_dir
    #     files = os.listdir(models_dir)
    #     for index in range(len(files)):
    #         files[index] = os.path.join(models_dir, files[index])

    #     # get the latest folder
    #     latest = max(files, key=os.path.getctime)
    #     # check if the folder has a checkpoint
    #     i = 1
    #     print(os.path.join(latest, f"mario_net_{i}.chkpt"))
    #     print(os.path.exists(os.path.join(latest, f"mario_net_{i}.chkpt")))
    #     if os.path.exists(os.path.join(latest, f"mario_net_{i}.chkpt")):
    #         while os.path.exists(os.path.join(latest, f"mario_net_{i+1}.chkpt")):
    #             i += 1
    #         print(
    #             f"Loading model from {os.path.join(latest, f'mario_net_{i}.chkpt')}")
    #         exit(0)
    #         mario.load(os.path.join(latest, f"mario_net_{i}.chkpt"))


    logger = MetricLogger(save_dir )

    try:
        episodes = 40000
        for e in range(episodes):

            state = env.reset()

            # Play the game!
            while True:

                # Run agent on the state
                action = mario.act(state)

                # Agent performs action
                next_state, reward, done, trunc, info = env.step(action)

                if not args.no_gui:
                    env.render()

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
                    break

            logger.log_episode()

            if (e % 20 == 0) or (e == episodes - 1):
                logger.record(episode=e, epsilon=mario.exploration_rate,
                            step=mario.curr_step)

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
    parser.add_argument("--mode", default="train", type=str, help="train | test")
    parser.add_argument("--load_dir", default=None, type=str, help="Path to load a model from if not in the default location")
    parser.add_argument("--save_dir", default=None, type=str, help="Path to save the model if not in the default location")
    parser.add_argument("--no_gui", action='store_true')

    args = parser.parse_args()

    main(args)
