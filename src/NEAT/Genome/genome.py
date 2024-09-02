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

    # Delete Genome
    lib.DeleteGenome.argtypes = [ctypes.c_void_p]
    lib.DeleteGenome.restype = None

    # Copy Genome
    lib.CopyGenome.argtypes = [ctypes.c_void_p]
    lib.CopyGenome.restype = ctypes.c_void_p

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

    # Print Genome
    lib.PrintGenomeInfo.argtypes = [ctypes.c_void_p]
    lib.PrintGenomeInfo.restype = None

    @property
    def fitness(self):
        return self.lib.GetFitness(self.genome)

    def __init__(self, name):
        self.genome = self.lib.NewGenome(name.encode('utf-8'))

    def __del__(self):
        self.lib.DeleteGenome(self.genome)

    def copy(self):
        g = Genome("test string")
        g.genome = self.lib.CopyGenome(self.genome)
        return g

    def new_genome(self, num_inputs, num_outputs):
        self.lib.InitGenome(self.genome, num_inputs, num_outputs)

    def load(self, filename):
        self.lib.LoadGenome(self.genome, filename.encode('utf-8'))

    def save(self, filename):
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

    def print_genome_info(self):
        self.lib.PrintGenomeInfo(self.genome)


def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (32, 32))
    img = img.flatten()
    c_floats = (ctypes.c_float * len(img))(*img)
    return c_floats
