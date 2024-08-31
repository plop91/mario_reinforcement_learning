from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import cv2
import os
import itertools
import numpy as np

from utils import *

import multiprocessing as mp

import warnings
warnings.filterwarnings("ignore")


def game_process(queue):
    print("Starting main process")
    print("Queue: ", queue)

    env = gym_super_mario_bros.make(
        'SuperMarioBros-v0', render_mode='rgb_array', apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    done = True
    i = 0
    while True:
        if done:
            state = env.reset()
            i = 0

        next_state, reward, done, trunc, info = env.step(
            env.action_space.sample())
        if queue.qsize() < 5:
            frame = env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            queue.put(frame)
        # TODO: remove test code
        # frame = env.render()
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.imshow(f"test{str(os.getpid())}", frame)
        # cv2.waitKey(1)
        # End test code
        i += 1

    env.close()


def tile_frames(frames: list):
    if len(frames) <= 0:
        return None

    # place the frames in a grid that is 4 frames wide and max height of 4 frames
    frame = None
    x = 0
    y = 0
    for i in range(len(frames)):
        if y >= 4:
            break
        row = None
        if x == 0:
            row = frames[i]
        else:
            if x < 3:
                frame = np.hstack((frame, frames[i]))
                x += 1
            else:
                if frame is None:
                    frame = row
                else:
                    frame = np.vstack((frame, row))
                x = 0
                y += 1
    return frame


def imgmatrix(frames):
    """
    https://gist.github.com/pgorczak/95230f53d3f140e4939c
    """
    w = 4
    h = 4
    n = w*h
    if len(frames) > n:
        imgs = frames[:n]
    elif len(frames) < n:
        imgs = frames + [np.zeros_like(frames[0]) for _ in range(n - len(frames))]
    else:
        imgs = frames

    if any(i.shape != imgs[0].shape for i in imgs[1:]):
        raise ValueError('Not all images have the same shape.')

    img_h, img_w, img_c = imgs[0].shape

    m_x = 0
    m_y = 0
    # if args.margin is not None:
    #     margin = args.margin[0]
    #     if '.' in margin:
    #         m = float(margin)
    #         m_x = int(m*img_w)
    #         m_y = int(m*img_h)
    #     else:
    #         m_x = int(margin)
    #         m_y = m_x

    imgmatrix = np.zeros((img_h * h + m_y * (h - 1),
                          img_w * w + m_x * (w - 1),
                          img_c),
                         np.uint8)

    imgmatrix.fill(255)

    positions = itertools.product(range(w), range(h))
    for (x_i, y_i), img in zip(positions, imgs):
        x = x_i * (img_w + m_x)
        y = y_i * (img_h + m_y)
        imgmatrix[y:y+img_h, x:x+img_w, :] = img

    return imgmatrix


def display_process(queues: list):
    print("Starting display process")
    print("Number of queues: ", len(queues))
    print("queues: ", queues)

    # cv2.startWindowThread()
    cv2.namedWindow("Mario mp test", cv2.WINDOW_NORMAL)

    if len(queues) <= 0:
        print("No queues to display")
        exit(-1)

    while True:
        # if any queue is empty restart the loop until all queues have frames
        for q in queues:
            if q.empty():
                continue
        # create and populate a list of frames
        # frame = None
        # for i in range(len(queues)):
        #     if frame is None:
        #         frame = queues[i].get()
        #     else:
        #         frame = np.hstack((frame, queues[i].get()))

        frame = imgmatrix([q.get() for q in queues])

        cv2.imshow("Mario mp test", frame)
        cv2.waitKey(1)
        # for i in range(len(queues)):
        #     if not queues[i].empty():
        #         frame = queues[i].get()
        #         cv2.imshow(f"test{str(i)}", frame)
        #         cv2.waitKey(1)


if __name__ == "__main__":
    print("Starting")

    num_envs = 32
    queues = []

    for i in range(num_envs):
        conn = mp.Queue()
        queues.append(conn)

    dp = mp.Process(target=display_process, args=(queues,))
    mario_processes = []

    for i in range(num_envs):
        mario_processes.append(mp.Process(
            target=game_process, args=(queues[i],)))

    try:
        dp.start()
        for p in mario_processes:
            p.start()

        dp.join()
        for p in mario_processes:
            p.join()
    except KeyboardInterrupt:
        print("Keyboard interrupt")
        dp.terminate()
        for p in mario_processes:
            p.terminate()
        cv2.destroyAllWindows()
