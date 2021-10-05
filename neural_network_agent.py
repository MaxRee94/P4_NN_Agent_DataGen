# Max van der Ree (6241638)

import random
import os
import copy
import numpy as np

import tensorflow as tf
from tensorflow import keras


def preprocess_board(board):
    preprocessed = []
    for tile in board.flatten():
        player_nodes = [(float(tile==0)), float(tile==1), float(tile==2)]
        preprocessed += player_nodes

    return preprocessed


class NNAgent:
    def __init__(self, id, verbosity="info", blueprint=None, compilation_settings=None, mode="play", saved_model_path=None):
        self.id = id
        self.verbosity = verbosity
        self.nn = NeuralNet(
            verbosity, mode, blueprint=blueprint, compilation_settings=compilation_settings, saved_model_path=saved_model_path
        )

    def preprocess_position(self, board, position):
        board = board.copy()
        board[position] = 1
        board = preprocess_board(board)
        active_player_input = [1.0, 0.0]

        preprocessed_position = board + active_player_input
        # print("preprocessed position:", preprocessed_position)

        return preprocessed_position

    def preprocess(self, free_positions, board):
        _input = []
        for position in free_positions:
            _input.append(self.preprocess_position(board, position))

        return _input

    def make_move(self, game):
        free_positions = game.board.free_positions()
        nn_input = self.preprocess(free_positions, game.board.board)
        hero_win_probabilities = self.nn.predict(nn_input)
        best_move_index = np.argmax(hero_win_probabilities)
        best_move = free_positions[best_move_index]

        return best_move

    def __str__(self):
        return f'Player {self.id} (NNAgent)'


class NeuralNet:

    checkpoint_default_path = os.path.join(os.path.dirname(__file__), "nn_model.ckpt")

    def __init__(
        self,
        verbosity,
        mode,
        blueprint=[
            {"width": 77, "activation": None},
            {"width": 1024, "activation": "relu"},
            {"width": 1024, "activation": "relu"},
            {"width": 3, "activation": "relu"},
        ],
        compilation_settings={
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "metrics": ["accuracy"],
        },
        saved_model_path=None,
    ):
        self.mode = mode
        if saved_model_path:
            self.saved_model_path = saved_model_path
        else:
            self.saved_model_path = self.get_saved_model_path()

        self.verbosity = verbosity
        if mode in ["play", "test"]:
            self.load_model()
        else:
            self.cp_callback = self.get_checkpoint_callback()
            architecture = self.get_architecture(blueprint)
            self.model = keras.Sequential(architecture)
            self.compile(self.model, compilation_settings)

    def get_saved_model_path(self):
        if self.mode in ["train", "test"]:
            i = 1
            output_root = os.path.join(os.path.dirname(__file__), "models")
            while True:
                _checkpoint_folder = os.path.join(output_root, "nn_model_v" + str(i).zfill(3) + ".ckpt")
                if os.path.exists(_checkpoint_folder):
                    if self.mode == "test" and not os.listdir(_checkpoint_folder):
                        checkpoint_folder = _checkpoint_folder
                    i += 1
                    checkpoint_folder = _checkpoint_folder
                else:
                    if self.mode == "train":
                        checkpoint_folder = _checkpoint_folder
                        os.mkdir(checkpoint_folder)
                        print("\nSaving model to:", checkpoint_folder)
                    break
            saved_model_path = os.path.join(checkpoint_folder, "nn_model.ckpt")
        else:
            saved_model_path = os.path.join(os.path.dirname(__file__), "nn_model.ckpt")

        return saved_model_path

    def get_architecture(self, blueprint):
        architecture = []
        print('blueprint:', blueprint)
        for layer_cfg in blueprint:
            layer = keras.layers.Dense(layer_cfg["width"], activation=layer_cfg["activation"])
            architecture.append(layer)

        return architecture

    def compile(self, model, compilation_settings):
        model.compile(**compilation_settings)

    def get_checkpoint_callback(self):
        callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.saved_model_path,
            verbose=1
        )
        return callback

    def predict(self, free_positions):
        win_probabilities = self.model.predict(free_positions)
        hero_win_probabilities = [prob[1] for prob in win_probabilities]

        return hero_win_probabilities

    def load_model(self):
        if self.verbosity == "debug":
            print("Loading model from:", self.saved_model_path)
        self.model = keras.models.load_model(self.saved_model_path, compile=True)

    def train(self, training_input, training_labels, training_settings={"epochs": 5}):
        result = self.model.fit(training_input, training_labels, callbacks=[self.cp_callback], **training_settings)

        return result

    def test(self, test_input, test_labels, print_result=True):
        test_loss, test_acc = self.model.evaluate(test_input, test_labels, verbose=1)

        if print_result:
            print("\nTest accuracy:", test_acc)
            print("Test loss:", test_loss)

        return test_acc
