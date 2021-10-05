# Max van der Ree (6241638)

from types import SimpleNamespace
import random, copy, time
import numpy as np

import game as g
from random_agent import RandomAgent


class BanditAgent:
    def __init__(self, iterations, id, verbosity="info", epsilon=99999999, initials=99999999):
        self.iterations = iterations
        self.id = id
        self.strategy = EGreedy(iterations, verbosity, epsilon=epsilon, initials=initials)

    def make_move(self, game):
        move = self.strategy.get_move(game)

        return move

    def __str__(self):
        return f'Player {self.id} (BanditAgent)'


class EGreedy:

    epsilon = 0.5
    initial_estimate = 0.5

    def __init__(self, iterations, verbosity, epsilon, initials):
        if not epsilon == 99999999:
            self.epsilon = epsilon  # Overwrite value if supplied through cli
        if not initials == 99999999:
            self.initial_estimate = initials  # Overwrite value if supplied through cli

        self.iterations = iterations
        self.verbosity = verbosity

    def get_greedy_move(self, game, estimates):
        unavailable_positions = game.board.board != 0
        _estimates = estimates.copy()
        _estimates[unavailable_positions] = -1
        move = np.unravel_index(np.argmax(_estimates), _estimates.shape)

        return move

    def select_move(self, game, estimates, force_greed=False):
        choose_greedy = force_greed or random.random() > self.epsilon
        if choose_greedy:
            return self.get_greedy_move(game, estimates)
        else:
            return game.board.random_free()

    def get_estimates(self, game):
        estimates = np.zeros((game.board.shape[0], game.board.shape[1]))
        estimates[:] = self.initial_estimate
        counts = np.zeros((game.board.shape[0], game.board.shape[1]))
        _iterations = self.iterations
        while _iterations > 0:
            # Determine which move to explore next
            move = self.select_move(game, estimates)

            # Evaluate the chosen move
            rollout = Simulation(game, move).rollout()

            # +1 the count of the chosen move (since this move now has +1 rollouts)
            counts[move[0], move[1]] += 1

            # Update the position of the chosen move with the found value
            self.update_estimate(estimates, move, rollout, counts)

            _iterations -= 1

        if self.verbosity == "debug":
            print("estimates: \n", estimates)
            print("counts:\n", counts)

        return estimates

    def update_estimate(self, estimates, move, rollout, counts):
        new_value = (estimates[move] * (counts[move] - 1) + rollout) / counts[move]
        estimates[move[0], move[1]] = new_value

    def get_move(self, game):
        estimates = self.get_estimates(game)
        move = self.select_move(game, estimates, force_greed=True)

        return move


class Simulation:

    hero = 1
    opponent = 2

    def __init__(self, game, move):
        self.move = move
        self.objectives = game.objectives
        self.players = [RandomAgent(self.opponent), RandomAgent(self.hero)]  # the order is important (opponent should do the next move)
        self.board_original = copy.deepcopy(game.board)
        self.game = game

    def reset(self):
        self.board = copy.deepcopy(self.board_original)
        self.board.place(self.move, self.hero)

        self.game = g.Game.from_board(
            self.board, self.objectives, self.players, False
        )

    def rollout(self):
        self.reset()
        if self.game.victory(self.move, self.hero):
            winner = SimpleNamespace(id=self.hero)
        else:
            winner = self.game.play()

        if winner is None:
            result = 0.5
        elif winner.id == self.hero:
            result = 1
        else:
            result = 0

        return result
