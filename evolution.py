import random
from tensorflow import keras
import numpy as np
import os
import time

from neural_network_agent import NNAgent


class Evolver():

    generation_size = 20
    survival_rate = 0.5
    mutation_rate = 0.1

    def __init__(self, training_input, training_labels, test_input, test_labels):
        self.evaluation_data = (training_input, training_labels, test_input, test_labels)
        self.generation = self.create_initial_generation()

    def create_blueprint(self):
        pass

    def create_initial_generation(self):
        generation = {}
        for i in range(self.generation_size):
            blueprint = self.create_blueprint()
            compilation_settings = self.create_compilation_settings()
            training_settings = self.create_training_settings()
            individual = Individual(blueprint, compilation_settings, training_settings)
            generation[i] = individual

        return generation

    def select(self):
        fitness_scores = np.zeros((1, self.generation_size))
        for id, individual in self.generation.items():
            fitness_scores[0, id] = individual.evaluate(**self.evaluation_data)


class Individual():
    def __init__(self, blueprint, compilation_settings, training_settings, verbosity="debug"):
        self.blueprint = blueprint
        self.compilation_settings = compilation_settings
        self.training_settings = training_settings
        self.nn = NNAgent(
            1, verbosity, self.blueprint, self.compilation_settings, load_weights=False, mode="train"
        ).nn

    def evaluate(self, training_input, training_labels, test_input, test_labels):
        training_settings = trial["training_settings"]
        self.nn.train(training_input, training_labels, training_settings)
        print("\nFinished training. Beginning test...")
        accuracy = self.nn.test(test_input, test_labels)


def evolve(training_input, training_labels, test_input, test_labels):
    evolver = Evolver(training_input, training_labels, test_input, test_labels)

