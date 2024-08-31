import ctypes
import subprocess
import os

import cv2

if not os.path.exists('./libgenome.so'):
    if os.path.exists('./Makefile'):
        subprocess.run(['make'])
    else:
        raise Exception('Makefile not found')


class Genome:
    lib = ctypes.cdll.LoadLibrary('./libgenome.so')

    # New Genome
    lib.NewGenome.argtypes = [ctypes.c_char_p]
    lib.NewGenome.restype = ctypes.c_void_p

    # Init new genome
    lib.InitGenome.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.InitGenome.restype = None

    # Load Genome
    lib.LoadGenome.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.LoadGenome.restype = None

    # Save Genome
    lib.SaveGenome.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.SaveGenome.restype = None

    # Mutate Genome
    lib.MutateGenome.argtypes = [ctypes.c_void_p]
    lib.MutateGenome.restype = None

    # Crossover Genome
    lib.CrossoverGenome.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.CrossoverGenome.restype = None

    # Feed Forward Genome
    lib.FeedForwardGenome.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.FeedForwardGenome.restype = ctypes.c_int

    # Set Name
    lib.SetName.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.SetName.restype = None

    # Get Name
    lib.GetName.argtypes = [ctypes.c_void_p]
    lib.GetName.restype = ctypes.c_char_p

    # Set Fitness
    lib.SetFitness.argtypes = [ctypes.c_void_p, ctypes.c_float]
    lib.SetFitness.restype = None

    # Get Fitness
    lib.GetFitness.argtypes = [ctypes.c_void_p]
    lib.GetFitness.restype = ctypes.c_float

    @property
    def fitness(self):
        return self.lib.GetFitness(self.genome)

    def __init__(self, name):
        self.genome = self.lib.NewGenome(name.encode('utf-8'))

    def __del__(self):
        self.lib.DeleteGenome(self.genome)

    def new_genome(self, num_inputs, num_outputs):
        self.lib.InitGenome(self.genome, num_inputs, num_outputs)

    def load_genome(self, filename):
        self.lib.LoadGenome(self.genome, filename.encode('utf-8'))

    def save_genome(self, filename):
        self.lib.SaveGenome(self.genome, filename.encode('utf-8'))

    def mutate(self):
        self.lib.MutateGenome(self.genome)

    def crossover(self, other):
        self.lib.CrossoverGenome(self.genome, other.genome)

    def feed_forward(self, inputs):
        return self.lib.FeedForwardGenome(self.genome, inputs)

    def set_name(self, name):
        self.lib.SetName(self.genome, name.encode('utf-8'))

    def get_name(self):
        return self.lib.GetName(self.genome).decode('utf-8')

    def set_fitness(self, fitness):
        self.lib.SetFitness(self.genome, fitness)

    def get_fitness(self):
        return self.fitness


def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (32, 32))
    img = img.flatten()
    c_floats = (ctypes.c_float * len(img))(*img)
    return c_floats


if __name__ == '__main__':
    from nes_py.wrappers import JoypadSpace
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
    try:
        from utils import *
    except ImportError:
        # add ../.. to sys.path
        import sys
        sys.path.append('../..')
        from utils import *
    import warnings
    warnings.filterwarnings("ignore")

    try:
        gui = False

        env = gym_super_mario_bros.make(
            'SuperMarioBros-1-1-v0', render_mode='rgb_array', apply_api_compatibility=True)
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        env = SkipFrame(env, skip=4)

        num_pop = 10
        print(f'Creating population of {num_pop} genomes')
        population = []
        for i in range(num_pop):
            g = Genome("test string")
            g.new_genome(1024, 10)
            population.append(g)

        print('Population created - starting evolution')
        for i in range(num_pop):
            last_state = np.zeros((32, 32, 3), dtype=np.uint8)
            env.reset()
            score = 0
            while True:
                c_floats = process_image(last_state)
                population[i].feed_forward(c_floats)
                next_state, reward, done, trunc, info = env.step(
                    env.action_space.sample())
                if gui:
                    cv2.imshow('SuperMario', cv2.cvtColor(next_state, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                score += reward
                if done or trunc:
                    break
                last_state = next_state
            population[i].set_fitness(score)
            print(f'Genome {i} fitness: {population[i].get_fitness()}')

        # Sort population by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        # print top 5 genomes
        for i in range(5):
            print(f'Genome {i} fitness: {population[i].get_fitness()}')

    except KeyboardInterrupt:
        pass
    finally:
        if os.path.exists('./libgenome.so'):
            os.remove('./libgenome.so')
        if os.path.exists('./genome.o'):
            os.remove('./genome.o')
