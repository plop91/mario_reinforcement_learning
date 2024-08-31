"""
This file contains my notes on the NEAT algorithm and how I plan to implement it in this project.

My understanding of the neat algorithm:

1. start with a nn that has only the base inputs, base outputs and a 'bias' node in the input layer, then instantiate x copies of the nn with randomized values, these are called genomes.
2. calculate the fitness of the nn i.e. sum all of its rewards over a single run.
3. determine the two fittest members of the population.
4. pair the fittest members of the population and create a new set of genomes using both parents
5. randomly mutate some of the new population
6. Repeat from step 2

more Notes:
- the net should be able to evolve activation functions
"""

import random

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import cv2
import os
import itertools
import numpy as np
import copy
import multiprocessing as mp

from utils import *

import warnings
warnings.filterwarnings("ignore")

# env = gym_super_mario_bros.make(
#     'SuperMarioBros-1-2-v0', render_mode='rgb_array', apply_api_compatibility=True)
# env = JoypadSpace(env, COMPLEX_MOVEMENT)

# env = SkipFrame(env, skip=4)
# env = GrayScaleObservation(env)
# env = ResizeObservation(env, shape=84)
# env = FrameStack(env, num_stack=4)

# cv2.namedWindow("Mario neat test", cv2.WINDOW_NORMAL)

num_input_nodes = 3
num_output_nodes = 3
num_hidden_nodes = 2
example_nodes = {
    0:0.0,
    1:1.0,
    2:0.25,
    4:float("-inf"),
    5:float("-inf"),
    6:float("-inf"),
    7:float("-inf"),
    8:float("-inf"),
}
example_connections = {
    #id:[in_node_id, out_node_id, weight, enabled]
    0:[0, 3, 0.5, True],
    1:[1, 3, -0.25, True],
    2:[2, 3, 0.75, True],
}


def get_connections_to_node(id, connections):
    conns = []
    for conn in connections:
        if conn.out_node == id:
            conns.append(conn)
    return conns


class NewGenotype:

    def __init__(self) -> None:
        # id:[type, value] Note: type 0=input, 1=hidden, 2=output
        self.nodes = {}
        self.connections = {}  # id:[in_node, out_node, weight, enabled]
        self.input_node_ids = []  # list of input node ids
        self.output_node_ids = []  # list of output node ids
        self.fitness = float("-inf")
        self.max_node_id = 0
        self.max_connection_id = 0

    def new_genome(self, num_inputs, num_outputs):
        # create the bias node
        self.nodes[self.max_node_id] = [0, 1.0]
        self.max_node_id += 1

        # create the input nodes
        for i in range(num_inputs):
            self.nodes[self.max_node_id] = [0, 0.0]
            self.input_node_ids.append(i)
            self.max_node_id += 1

        # create the output nodes
        for i in range(num_outputs):
            self.nodes[self.max_node_id] = Node(i+num_inputs, "output")
            self.output_node_ids.append(i+num_inputs)
            self.max_node_id += 1

        # connect the bias node to all the output nodes
        for i in range(num_outputs):
            self.connections[self.max_connection_id] = [
                0, self.output_node_ids[i], random.uniform(-1, 1), True]
            self.max_connection_id += 1

    def feed_forward(self, state):

        def _node_ready(node_id, open_connections):
            for conn in open_connections:
                if self.connections[conn][1] == node_id:
                    return False
            return True
        
        def _get_connections_to_node(node_id):
            conns = []
            for conn_id in self.connections:
                if self.connections[conn_id][1] == node_id:
                    conns.append(node_id)
            return conns  

        # load the input nodes with the state
        for i in range(self.inputs):
            self.nodes[self.input_node_ids[i]].value = state[i]

        open_connections = list(self.connections.keys())

        while len(open_connections) > 0:
            conn_id = open_connections.pop(0)
            if not self.connections[conn_id][3]:
                continue
            # check if the from node is done
            if _node_ready(self.connections[conn_id][0], open_connections):
                conns = _get_connections_to_node()

            else:
                open_connections.append(conn_id)

        pass


class Neat:
    def __init__(self, num_inputs, num_outputs, max_population=100, show_gui=False, use_mp=False):
        """

        arg: inputs: number of inputs to the neural network (not including the bias node)
        arg: outputs: number of outputs from the neural network
        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.max_population = max_population
        self.population = []
        self.show_gui = show_gui
        self.use_mp = use_mp

    def run(self):
        g = Genotype(self.num_inputs, self.num_outputs)
        g.new_genome()

        self.population.append(g)
        for i in range(self.max_population-1):
            self.population.append(g.mutate())

        while True:
            self.evolve()

    def evolve(self):
        if self.use_mp:
            try:
                pool = mp.Pool(processes=os.cpu_count())
                foo = pool.map(self._evaluate_fitness,
                               range(len(self.population)))
                pool.close()
                for i in range(len(self.population)):
                    self.population[i].fitness = foo[i]
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                pool.close()
                raise KeyboardInterrupt

        else:
            # now we have a population of genomes, we can start the evolution process
            for genome in self.population:
                genome.fitness = genome.evaluate_fitness(
                    iterations=10, show_gui=self.show_gui)

        # find the two fittest genomes
        fittest_genomes = sorted(
            self.population, key=lambda x: x.fitness, reverse=True)[:2]
        print(
            f"Two fittest genomes: {fittest_genomes[0].fitness}, {fittest_genomes[1].fitness}")

        # TODO: log/save the fittest genomes

        # crossover the two fittest genomes
        new_genome = fittest_genomes[0].crossover(fittest_genomes[1])
        print(f"New genome: {new_genome}")

        self.population.clear()
        self.population.append(new_genome)
        for i in range(self.max_population-1):
            self.population.append(new_genome.mutate())

    def _evaluate_fitness(self, genome_index):
        return self.population[genome_index].evaluate_fitness(iterations=1, show_gui=self.show_gui)


class Genotype:

    def __init__(self, inputs, outputs):
        """

        arg: inputs: number of inputs to the neural network (not including the bias node)
        arg: outputs: number of outputs from the neural network
        """

        self.num_inputs = inputs
        self.num_outputs = outputs
        self.nodes = []  # a list of nodes
        self.input_nodes = []  # a list of input node ids
        self.hidden_nodes = []  # a list of hidden node ids
        self.output_nodes = []  # a list of output node ids
        self.connections = []  # a list of connections
        self.max_node_id = 0

        self.fitness = float("-inf")

    @property
    def id(self):
        h = 0
        for node in self.nodes:
            h += hash(node)
        for connection in self.connections:
            h += hash(connection)
        return h

    def new_genome(self):
        # first node is the bias node
        bias_node = Node(0, "input", 1.0)
        self.nodes.append(bias_node)
        self.input_nodes.append(bias_node.id)

        for i in range(self.num_inputs):
            input_node = Node(i+1, "input")
            self.nodes.append(input_node)
            self.input_nodes.append(input_node.id)

        for i in range(self.num_outputs):
            output_node = Node(i+self.inputs+1, "output")
            self.nodes.append(output_node)
            self.output_nodes.append(output_node.id)

        self.max_node_id = self.inputs + self.outputs + 1

        # connect the bias node to all the output nodes
        for i in range(self.outputs):
            self.connections.append(Connection(
                0, self.output_nodes[i], random.uniform(-1, 1)))

        for i in range(self.outputs):
            print(f"Output node {self.outputs[i]}")

    def mutate(self):
        """
        mutate the genome
        NOTE: the first pass will only mutate the weights of the connections
        """
        """
        mutate options:
        - adjust the weight of a connection :50%
        - add a new connection: 10%
        - add a new node: 10%
        - disable a connection: 10%
        - enable a connection: 10%
        - change an activation function: 10%
        """
        mutation = random.uniform(0, 1)
        new_genome = copy.deepcopy(self)
        try:
            if mutation < 0.5:
                # mutation 1: adjust the weight of a connection
                # print("Mutating weight")
                new_genome.connections[random.randint(
                    0, len(new_genome.connections)-1)].weight += random.uniform(-0.1, 0.1)
            elif mutation < 0.6:
                # mutation 2: add a new connection
                # print("Mutating connection")
                # TODO: check that the connection does not already exist
                # TODO: check that the connection does not create a cycle
                # TODO: check that the connection does not connect to an input node
                raise NotImplementedError
                pass
            elif mutation < 0.7:
                # mutation 3: add a new node
                # adding a node will disable a connection and create two new connections
                # print("Mutating node")
                raise NotImplementedError
                pass
            elif mutation < 0.8:
                # mutation 4: disable a connection
                # print("Mutating disable")
                # TODO: check that the connection is not already disabled
                # TODO: check that the connection is not the only connection to an output node
                raise NotImplementedError
                pass
            elif mutation < 0.9:
                # mutation 5: enable a connection
                # print("Mutating enable")
                # TODO: check that the connection is not already enabled
                # TODO: check that enabling the connection does not create a cycle
                raise NotImplementedError
                pass
            else:
                # mutation 6: change an activation function
                # print("Mutating activation")
                # TODO: check that the node is not an input or output node
                # TODO: check that the node is not the bias node
                # TODO: check that the node does not already use the activation function
                raise NotImplementedError
                pass
        except NotImplementedError:
            new_genome = copy.deepcopy(self)
            new_genome.connections[random.randint(
                0, len(new_genome.connections)-1)].weight += random.uniform(-0.1, 0.1)
        return new_genome

    def crossover(self, other) -> "Genotype":
        """
        Crossover this genome with another genome
        The genotype with the higher fitness should be the caller
        """
        return self

    def evaluate_fitness(self, iterations=10, show_gui=False):
        """
        run the mario game and accumulate the rewards
        """
        try:
            # TEMP: env will be made here now
            env = gym_super_mario_bros.make(
                'SuperMarioBros-1-2-v0', render_mode='rgb_array', apply_api_compatibility=True)
            env = JoypadSpace(env, COMPLEX_MOVEMENT)
            env = SkipFrame(env, skip=4)

            id = self.id
            print(f"Genome {id} Evaluating fitness")
            env.reset()
            running_reward = 0
            for i in range(iterations):
                print(f"Genome {id} Running iteration {i}")
                env.reset()
                reward = self.evaluate_single_run(env, show_gui=show_gui)
                running_reward += reward
                print(f"Genome {id} Iteration: {i} Reward: {reward}")
            fitness = running_reward / iterations
            print(f"Genome {id} fitness: {fitness}")
            env.close()
            return fitness
        except KeyboardInterrupt:
            env.close()
            raise KeyboardInterrupt

    def evaluate_single_run(self, env, show_gui=False):
        running_reward = 0
        last_state = np.zeros((240, 256, 3), dtype=np.uint8)

        while True:

            state = self.preprocess_state(last_state)
            action = self.feed_forward(state, env)

            state, reward, truncated, terminated, info = env.step(action)

            last_state = state
            running_reward += reward

            done = truncated or terminated
            if done or info["flag_get"]:
                break
            if show_gui:
                frame = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
                cv2.imshow("Mario neat test", frame)
                cv2.waitKey(1)
        return running_reward

    def preprocess_state(self, state: np.ndarray) -> np.ndarray:
        """
        convert the state which is a rgb image that is 240x256x3 to a 1d array of 0s and 1s
        """
        # TODO: find a numpy way to do this, most likely faster than cv2?
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        state = cv2.resize(state, (32, 32))
        return state.flatten()

    def feed_forward(self, state, env):
        # TODO: implement the feed forward algorithm

        # my attempt at the feed forward algorithm, the first pass will only consider the weights of the connections
        # the second pass will consider the activation functions of the nodes

        # first pass

        # goal given an input(state) the net should "activate" every connection in the net, however,
        # a connection can not be activated if all of the connections leading to the input node have not been activated

        # load the input nodes with the state
        for i in range(self.inputs):
            self.nodes[i+1].value = state[i]

        # what do we know?
        # that any node marked as an input node that has a connection leading to it should go first
        def get_node_with_id(id, nodes):
            for node in nodes:
                if node.id == id:
                    return node
            return None

        def check_if_node_is_ready(id, open_connections):
            for conn in open_connections:
                if conn.out_node == id:
                    return False
            return True

        # open_nodes = copy.deepcopy(self.nodes)
        # closed_nodes = []
        open_connections = copy.deepcopy(self.connections)
        closed_connections = []

        while len(open_connections) > 0:
            # for each connection in the open connections list, we dont know if the nodes they come from are in the closed list so check
            connection = open_connections.pop(0)
            if check_if_node_is_ready(connection.in_node, open_connections):
                # run the connection
                connection.value = get_node_with_id(
                    connection.in_node, self.nodes).value * connection.weight
                closed_connections.append(connection)
            else:
                # the node is not ready re add it to the queue
                open_connections.append(connection)
        # when there are no more connections the model is done running

        self.connections = closed_connections

        # determine which action to take and return that action
        max_value = float("-inf")
        max_index = 0
        for i in range(self.outputs):
            # get all connections to the output node
            conns = get_connections_to_node(
                self.output_nodes[i].id, self.connections)
            print(len(conns))
            value = 0
            for conn in conns:
                value += conn.value
            if value > max_value:
                max_value = value
                max_index = i
        print(f"Action: {max_index} with value {max_value}")
        return max_index

        # # TODO: replace test code
        # return env.action_space.sample()

    def __str__(self) -> str:
        return f"Genotype id: {self.id} with fitness {self.fitness}"

    def __repr__(self):
        return str(self)


class Node:
    def __init__(self, id, node_type, value: float = 0.0):
        """
        arg: id: the id of the node
        arg: node_type: the type of the node (input, output, hidden, activation)
        """
        self.id = id
        self.node_type = node_type
        self.value = value

    def __str__(self) -> str:
        return f"Node {self.id} of type {self.node_type}"

    def __repr__(self):
        return str(self)


class Connection:
    def __init__(self, in_node, out_node, weight, enabled=True):
        """

        arg: in_node: the id of the node the connection is coming from
        arg: out_node: the id of the node the connection is going to
        arg: weight: the weight of the connection
        """
        # self.id = 0
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.value = float("-inf")

    def __str__(self) -> str:
        return f"Connection from {self.in_node} to {self.out_node} with weight {self.weight} and enabled {self.enabled}"

    def __repr__(self):
        return str(self)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_gui", action="store_true")
    parser.add_argument("--use_mp", action="store_true")

    args = parser.parse_args()

    n = Neat(1024, len(COMPLEX_MOVEMENT), max_population=100,
             show_gui=args.show_gui, use_mp=args.use_mp)
    n.run()
