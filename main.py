#! /usr/bin/env -S python -u
import random
import argparse, time, cProfile
import numpy as np
import multiprocessing as mp
from collections import Counter
import copy
from itertools import starmap

from game import Game
from board import Board
from random_agent import RandomAgent
from neural_network_agent import NNAgent
from neural_network_agent_dimitris import NNAgent as DimitrisNNAgent

from bandit_agent import BanditAgent
from mcts_agent import Agent as MCTSAgent
from score_agent import Agent as ScoreAgent


split_threshold = 30
agents = [
    {"cls": RandomAgent, "args": []},
    {"cls": BanditAgent, "args": [10]},
    {"cls": BanditAgent, "args": [100]},
    {"cls": BanditAgent, "args": [200]},
    {"cls": MCTSAgent, "args": [250]},
    {"cls": DimitrisNNAgent, "args": []},
    {"cls": ScoreAgent, "args": []}
]


def get_agent_randomly(id=None):
    assert id, "Pass an id to the 'get agent randomly' function"
    agent_config = random.choice(agents)
    agent_cls = agent_config["cls"]
    args = agent_config["args"] + [id]
    agent = agent_cls(*args)

    return agent


def main(args):
    if args.split_games and args.games > split_threshold:
        total_games = args.games
        args.games = split_threshold

        finished_games = 0
        while total_games >= split_threshold:
            _main(args)
            total_games -= split_threshold
            finished_games += split_threshold
            print("Finished running", finished_games, "games.")

        args.games = total_games

    _main(args)


def _main(args):
    if args.input:
        data = read_games(args.input)
        # call some function to do your preprocessing and training here

    split_task = args.games > split_threshold
    if split_task:
        original_amount = args.games
        args.games = split_threshold

    work = []
    for i in range(args.games):
        if i % 10 == 0:
            print("adding game", i, "/", args.games, "...")
        if args.output:
            hero = get_agent_randomly(id=1)
        else:
            hero = NNAgent(1)

        opponent = get_agent_randomly(id=2)
        # swap order every game
        if i % 2 == 0:
            players = [hero, opponent]
        else:
            players = [opponent, hero]

        work.append((args.size,
                    read_objectives(args.objectives),
                    players,
                    args.output,
                    args.print_board))

    print("\nrunning", len(work), "games...")
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

    if split_task:
        args.games = original_amount - split_threshold
        main(args)

def play_game(boardsize, objectives, players, output, print_board = None):
    game = Game.new(boardsize, objectives, players, print_board == 'all')

    if output:
        with open(output, 'a+') as outfile:
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

    parser.add_argument('objectives',
        help = 'The name of a file containing the objective shapes. The file '
        'should contain a rectangle with x on positions that should be '
        'occupied, and dots on other positions. Separate objective shapes '
        'should be separated by a blank line.')

    parser.add_argument('--split_games', action="store_true")

    args = parser.parse_args()
    #cProfile.run('main(args)')
    main(args)
