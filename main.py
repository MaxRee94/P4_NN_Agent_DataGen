#! /usr/bin/env -S python -u
from game import Game
from board import Board
from random_agent import RandomAgent
from neural_network_agent import NNAgent, preprocess_board
import os
import argparse, time, cProfile
import numpy as np
import multiprocessing as mp
from collections import Counter
from itertools import starmap
import time
import constants
import evolution
import random
import json

import tensorflow as tf
from tensorflow import keras


def init_outfile():
    output_folder = r"E:\Documents\AI_for_Game_Technology\class_assignments\P3_NN_1_Agent\hyperparameter_trials"
    i = 1
    while True:
        output_file = os.path.join(output_folder, "v" + str(i).zfill(3) + ".json")
        if os.path.exists(output_file):
            i += 1
        else:
            break

    with open(output_file, "w") as f:
        f.write("{}")

    return output_file


def preprocess_move(move, boardshape):
    _move = np.zeros(boardshape)
    _move[move] = 1
    move = list(_move.flatten())

    return move


def preprocess(data, testsize, verbosity, usage_fraction):
    t_start = time.perf_counter()
    input = []
    labels = []
    train_size = int(len(data) * usage_fraction)
    print("Preprocessing data...")
    print("     Using", train_size, "/", len(data), "games to train (", usage_fraction * 100, "% of total)")

    printsize = int(train_size / 10)
    for i, game in enumerate(data):
        if verbosity == "debug" and i % printsize == 0:
            print("Preprocessing...", i / printsize * 10, "%")

        winner = game[0]
        game_labels = [float(winner==0), float(winner==1), float(winner==2)]
        states = game[1]
        for j, state in enumerate(states):
            active_player, move, board = state

            #active_player_input = [float(active_player==1), float(active_player==2)]
            board = preprocess_board(board)
            #state_input = board + active_player_input
            state_input = board

            input.append(state_input)
            labels.append(game_labels)

        if i == train_size:
            break

    split_index = int(testsize * len(input))
    print("training input example:", input[0])
    print("training labels example:", labels[0])
    training_input = tf.constant(input[split_index:], dtype=tf.float64)
    training_labels = tf.constant(labels[split_index:], dtype=tf.float64)
    test_input = tf.constant(input[:split_index], dtype=tf.float64)
    test_labels = tf.constant(labels[:split_index], dtype=tf.float64)

    json_content = {
        "train_input": input[split_index:],
        "train_label": labels[split_index:],
        "test_input": input[:split_index],
        "test_labels": labels[:split_index],
    }
    with open(r"E:\Documents\AI_for_Game_Technology\class_assignments\P3_NN_1_Agent\preprocessed.json", "w") as f:
        import json
        json.dump(json_content, f)

    print(f"\nFinished data preprocessing. Time taken: {time.perf_counter() - t_start:1f} seconds.")

    return training_input, training_labels, test_input, test_labels


def load_from_json(json_path):
    t_start = time.perf_counter()
    print("\nLoading from json...")
    with open(json_path, "r") as f:
        content = json.load(f)

    print(
        "json content training input:", len(content["train_input"]), len(content["train_label"]),
        len(content["test_input"]), len(content["test_labels"])
    )
    print("Finished loading from json. Time taken:", time.perf_counter() - t_start, "seconds.\n")

    return list(content.values())


def save_trial(trial, accuracy):
    with open(output_file, "r") as f:
        data = json.load(f)

    trial["compilation_settings"]["optimizer"] = str(trial["compilation_settings"]["optimizer"])

    data[random.random()] = {"trial:": trial, "accuracy": accuracy}

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


def log_max_accuracy(log):
    max_acc = max(list(log.keys()))
    max_trial = log[max_acc]
    print(f"\nCurrent best result has {max_acc*100} % accuracy. Corresponding trial:\n", max_trial)


def save_data_portion(training_input, training_labels, test_input, test_labels, split_index, portion_fraction=0.1):
    portion_size = int(len(training_input) / (1/portion_fraction))
    print("portion size:", portion_size)
    json_content = {
        "train_input": training_input[:portion_size],
        "train_label": training_labels[:portion_size],
        "test_input": test_input,
        "test_labels": test_labels,
    }
    with open(r"E:\Documents\AI_for_Game_Technology\class_assignments\P3_NN_1_Agent\preprocessed_onetenth.json", "w") as f:
        import json
        json.dump(json_content, f)
    print("\ndone.")

    return training_input[split_index:int(len(training_input)/(1/portion_size))]


def main(args):
    if args.input:
        global output_file
        output_file = init_outfile()
        if args.from_json:
            training_input, training_labels, test_input, test_labels = load_from_json(args.from_json)
            # split_index = int(args.testsize * len(training_input))
            # training_input = training_input[split_index:]
            # training_labels = training_labels[split_index:]
        else:
            data = read_games(args.input)
            training_input, training_labels, test_input, test_labels = preprocess(
                data, args.testsize, args.verbosity, args.usage_fraction
            )
        # training_input = save_data_portion(training_input, training_labels, test_input, test_labels, int(args.testsize * len(training_input)), portion_fraction=0.1)
        # print("training input:", training_input[0])
        # return

        log = {}
        for trial in constants.trials:
            nn = NNAgent(
                1,
                verbosity=args.verbosity,
                blueprint=trial["blueprint"],
                compilation_settings=trial["compilation_settings"],
                mode="train"
            ).nn
            print("using learnrate:", trial["learnrate"])
            print("using", len(training_input), "samples to train.")
            print("\nNeural net initialized. Beginning training...\n")
            training_settings = trial["training_settings"]
            if args.epochs:
                training_settings["epochs"] = args.epochs
                print("Overwriting epochs in constants.py with user-defined epochs number:", args.epochs)

            _ = nn.train(training_input, training_labels, training_settings)
            print("\nFinished training. Beginning test...")
            accuracy = nn.test(test_input, test_labels)
            save_trial(trial, accuracy)
            log[accuracy] = trial
            log_max_accuracy(log)
        # evolution.evolve(training_input, training_labels, test_input, test_labels)

        print("\n--- Finished all trials.")

        return

    work = []
    for i in range(args.games):
        # swap order every game
        #nn_agent = NNAgent(1, mode="test", saved_model_path=args.saved_model)
        nn_agent = NNAgent(1)
        if i % 2 == 0:
            players = [nn_agent, RandomAgent(2)]
        else:
            players = [RandomAgent(2), nn_agent]

        work.append((args.size,
                     read_objectives(args.objectives),
                     players,
                     args.output,
                     args.print_board))

    start = time.perf_counter()

    # the tests can be run in parallel, or sequentially
    # it is recommended to only use the parallel version for large-scale testing
    # of your agent, as it is harder to debug your program when enabled
    if args.parallel == None or args.output != None:
        results = starmap(play_game, work)
    else:
        # you probably shouldn't set args.parallel to a value larger than the
        # number of cores on your CPU, as otherwise agents running in parallel
        # may compete for the time available during their turn
        with mp.Pool(args.parallel) as pool:
            results = pool.starmap(play_game, work)

    stats = Counter(results)
    end = time.perf_counter()

    print(f'Total score {stats[1]}/{stats[2]}/{stats[0]}.')
    print(f'Total time {end - start} seconds.')

def play_game(boardsize, objectives, players, output, print_board = None):
    game = Game.new(boardsize, objectives, players, print_board == 'all')

    if output:
        with open(output, 'a') as outfile:
            print(boardsize, file = outfile)
            winner = game.play(outfile)
            print(f'winner={winner.id if winner else 0}', file = outfile)
    else:
        winner = game.play()

    if print_board == 'final':
        game.print_result(winner)

    return 0 if winner == None else winner.id

def read_objectives(filename):
    with open(filename) as file:
        lines = [line.strip() for line in file]

    i = 0
    shapes = []
    while i < len(lines):
        shape = []

        # shapes are separated by blank lines
        while i < len(lines) and lines[i].strip() != '':
            shape_line = []
            for char in lines[i].strip():
                shape_line.append(char == 'x')
            shape.append(shape_line)
            i += 1

        shapes.append(np.transpose(np.array(shape)))
        i += 1

    return shapes

def read_games(filename):
    print(f"Reading data from '{filename}'...")
    t_start = time.perf_counter()
    with open(filename) as file:
        lines = list(file)

        games = []

        i = 0
        while i < len(lines):
            game = []
            boardsize = int(lines[i])
            i += 1

            while not lines[i].startswith('winner'):
                turn = int(lines[i])
                i += 1
                move = [int(x) for x in lines[i].split(',')]
                i += 1
                board = np.zeros((boardsize, boardsize), dtype = int)
                for y in range(boardsize):
                    row = lines[i].split(',')
                    for x in range(boardsize):
                        board[(x, y)] = int(row[x])
                    i += 1

                game.append((turn, move, board))

            winner = int(lines[i].split('=')[1])
            games.append((winner, game))

            i += 1

    print(f"\nFinished loading data. Time taken: {time.perf_counter() - t_start:1f} seconds.")

    return games

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type = int, default = 10,
        help = 'The size of the board.')

    parser.add_argument('--games', type = int, default = 1,
        help = 'The number of games to play.')

    parser.add_argument('--time', type = int, default = 10,
        help = 'The allowed time per move, in milliseconds.')

    parser.add_argument('--print-board', choices = ['all', 'final'],
        help = 'Show the board state, either every turn or only at the end.')

    parser.add_argument('--parallel', type = int,
        help = 'Run multiple games in parallel. Only use for large-scale '
        'testing.')

    parser.add_argument('--output',
        help = 'Write training data to the given file.')

    parser.add_argument('--input',
        help = 'Read training data from the given file.')

    parser.add_argument("--testsize",
        type=float,
        default=0.02,
        help = "size of test dataset (as fraction of total available data).")

    parser.add_argument(
        "--verbosity",
        type=str,
        default="info",
    )

    parser.add_argument(
        "--from_json",
        type=str,
        default=""
    )

    parser.add_argument(
        "--usage_fraction",
        "-tf",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="play",
    )

    parser.add_argument(
        "--saved_model",
        type=str,
        default=None
    )

    parser.add_argument('objectives',
        help = 'The name of a file containing the objective shapes. The file '
        'should contain a rectangle with x on positions that should be '
        'occupied, and dots on other positions. Separate objective shapes '
        'should be separated by a blank line.')

    args = parser.parse_args()
    #cProfile.run('main(args)')
    main(args)
