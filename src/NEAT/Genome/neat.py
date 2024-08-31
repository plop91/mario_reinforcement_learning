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
import warnings
warnings.filterwarnings("ignore")


class Neat:
    def __init__(self, max_population=100, gui=False):
        self.max_population = max_population  # Number of genomes in the population
        self.population = []  # List of genomes
        self.gui = gui

    def create_population(self):
        for i in range(self.max_population):
            g = Genome(f"initial genome: {i}")
            g.new_genome(1024, 10)
            self.population.append(g)
        print(f'Created population of {self.max_population} genomes')

    def evolve(self):
        print('Evolving population...')
        env = gym_super_mario_bros.make(
            'SuperMarioBros-1-1-v0', render_mode='rgb_array', apply_api_compatibility=True)
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        env = SkipFrame(env, skip=4)

        for i in range(self.max_population):
            print(f'Genome {i} Evaluating')
            last_state = np.zeros((32, 32, 3), dtype=np.uint8)
            env.reset()
            score = 0
            while True:
                c_floats = process_image(last_state)
                self.population[i].feed_forward(c_floats)
                next_state, reward, done, trunc, info = env.step(
                    env.action_space.sample())
                if self.gui:
                    cv2.imshow('SuperMarioBros-1-1', cv2.cvtColor(
                        next_state, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                score += reward
                if done or trunc:
                    break
                last_state = next_state
            self.population[i].set_fitness(score)
            print(f'Genome {i} fitness: {self.population[i].get_fitness()}')

        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # print top 5 genomes
        print('Top 5 genomes:')
        for i in range(5):
            print(f'Genome {i} fitness: {self.population[i].get_fitness()}')

        parent = self.population.pop(0)
        self.population = []
        self.population.append(parent)
        for i in range(self.max_population-1):
            child = parent.copy()
            child.mutate()
            self.population.append(child)
            

        


def main():

    try:
        gui = False

        neat = Neat(max_population=10, gui=gui)
        neat.create_population()
        neat.evolve()

    except KeyboardInterrupt:
        pass
    finally:
        if os.path.exists('./libgenome.so'):
            os.remove('./libgenome.so')
        if os.path.exists('./genome.o'):
            os.remove('./genome.o')


if __name__ == '__main__':
    main()
