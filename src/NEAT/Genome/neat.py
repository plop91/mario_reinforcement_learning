from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from genome import *

try:
    from utils import *
except ImportError:
    import sys
    sys.path.append('../..')
    from utils import *

import os
import shutil
import time

import warnings
warnings.filterwarnings("ignore")


class Neat:
    """
    What are the things the neat class should do?
    1. Create a population of genomes
    - This is important because we need to have a population of genomes to evolve
    2. Evolve the population
    - This is important because we need to evolve the population of genomes to improve the fitness of the genomes
        - This will include saving checkpoints of the population 
    3. Save the current population
    - This is important because we need to save the current population of genomes to a file so that we can load it later
    4. Load a population
    - This is important because we need to load a population of genomes from a file so that we can evolve it further


    
    """

    def __init__(self, max_population=100, gui=False):
        self.max_population = max_population  # Number of genomes in the population
        self.population = []  # List of genomes
        self.gui = gui

    def create_population(self):
        for i in range(self.max_population):
            g = Genome(f"initial genome: {i}")
            g.new_genome(1024, len(COMPLEX_MOVEMENT))
            self.population.append(g)
        print(f'Created population of {self.max_population} genomes')

    def evolve(self):
        print('Evolving population...')
        generation = 0
        while True:
            print(f'Generation {generation} started')
            env = gym_super_mario_bros.make(
                'SuperMarioBros-1-1-v0', render_mode='rgb_array', apply_api_compatibility=True)
            env = JoypadSpace(env, COMPLEX_MOVEMENT)
            env = SkipFrame(env, skip=4)

            print(f'Generation {generation} environment created')

            for i in range(self.max_population):
                print(f'Generation {generation} Genome {i} Evaluating')
                last_state = np.zeros((32, 32, 3), dtype=np.uint8)
                env.reset()
                score = 0
                while True:
                    c_floats = process_image(last_state)
                    self.population[i].feed_forward(c_floats)
                    next_state, reward, done, trunc, info = env.step(
                        env.action_space.sample())
                    if self.gui:
                        frame = cv2.cvtColor(next_state, cv2.COLOR_RGB2BGR)
                        # frame = write_neat_info_on_img(frame, generation, i, score)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        frame = cv2.resize(frame, (32, 32))
                        cv2.imshow('SuperMarioBros-1-1', frame)
                        cv2.waitKey(1)
                    score += reward
                    if done or trunc:
                        break
                    last_state = next_state
                self.population[i].set_fitness(score)
                print(
                    f'Generation {generation} Genome {i} fitness: {self.population[i].get_fitness()}')

            # Sort population by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)

            # print top 5 genomes
            print('Top 5 genomes:')
            for i in range(5):
                print(
                    f'Genome {i} fitness: {self.population[i].get_fitness()}')

            parent = self.population.pop(0)
            self.population = []
            self.population.append(parent)
            for i in range(self.max_population-1):
                child = parent.copy()
                child.mutate()
                self.population.append(child)
            generation += 1

    def save(self, path: str):
        # first create a directory to save the population
        if not os.path.exists(path):
            os.makedirs(path)

        # save Neat and population information to a file
        with open(os.path.join(path, 'neat.info'), 'w+') as f:
            # TODO: Save additional information about the population
            f.write(f'{self.max_population}\n')

        # save each genome in the population to a file
        os.mkdir(os.path.join(path, 'genomes'))
        for i in range(len(self.population)):
            os.mkdir(os.path.join(path, 'genomes', f'genome_{i}'))
            self.population[i].save(os.path.join(
                path, 'genomes', f'genome_{i}'))

        # zip the files and save them to a file
        shutil.make_archive(path, 'zip', path)

        # remove the directory
        shutil.rmtree(path)

    def load(self, filename):
        # unzip the file and extract the files to a directory
        shutil.unpack_archive(filename + ".zip", filename)

        # load Neat and population information from the file
        with open(os.path.join(filename, 'neat.info'), 'r') as f:
            self.max_population = int(f.readline())

        print(f'Info file read, {self.max_population} genomes')

        # load each genome in the population from a file
        self.population = []
        for i in range(self.max_population):
            g = Genome("test string")
            g.load(os.path.join(filename, 'genomes', f'genome_{i}'))
            self.population.append(g)

        print(f'Genomes loaded, {len(self.population)} genomes')

        # remove the directory
        shutil.rmtree(filename)


def main():

    try:
        gui = False

        neat = Neat(max_population=50, gui=gui)
        neat.create_population()
        # neat.evolve()
        start_save = time.time()
        neat.save('./p1')
        end_save = time.time()
        print(f'Save time: {end_save-start_save}')
        start_load = time.time()
        neat.load('./p1')
        end_load = time.time()
        print(f'Load time: {end_load-start_load}')
        exit(0)

    except KeyboardInterrupt:
        pass
    finally:
        if os.path.exists('./libgenome.so'):
            os.remove('./libgenome.so')
        if os.path.exists('./genome.o'):
            os.remove('./genome.o')


if __name__ == '__main__':
    main()
