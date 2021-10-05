# Max van der Ree (6241638)

from types import SimpleNamespace
import random, copy, time
import numpy as np
import math


class Agent:
    def __init__(self, id, verbosity="info"):
        self.id = id
        self.verbosity = verbosity

    def get_score(self, game, move):
        board = copy.deepcopy(game.board)
        board.place(move, 1)
        scorer = Scorer(board, game.objectives, game.players, None)

        return scorer.get_score(move, 1, 2)

    def get_scores(self, game):
        scores = np.zeros([game.board.shape[0], game.board.shape[1]])
        scores -= 1  # Set default scores to -1 to avoid selecting an already-occupied tile
        for move in game.board.free_positions():
            score = self.get_score(game, move)
            scores[tuple(move)] = score

        if self.verbosity == "debug":
            print("\nScores:\n", np.transpose(scores), "\n")

        return scores

    def make_move(self, game):
        scores = self.get_scores(game)
        move = np.unravel_index(np.argmax(scores), scores.shape)
        if self.verbosity == "debug":
            print("chosen move:", move)

        return move

    def __str__(self):
        return f'Player {self.id} (Score Agent)'


class Scorer():

    win_factor = 100
    near_win_factor = 1

    def __init__(self, board, objectives, players, print_board):
        self.board = board
        self.players = players
        self.print_board = print_board
        self.objectives = [objective.astype(int) for objective in objectives]

    def get_components(self, shape):
        components = []
        for x in range(shape.shape[0]):
            for y in range(shape.shape[1]):
                if shape[x, y]:
                    components.append((x, y))

        return components

    def get_incomplete_shapes(self):
        incomplete_shapes = []
        for shape in self.objectives:
            components = self.get_components(shape)
            permutations = []
            for component in components:
                _shape = shape.copy()
                _shape[component] = -1
                permutations.append(_shape)

            incomplete_shapes += permutations

        return incomplete_shapes

    def get_fits_count(self, move, player, objectives):
        xh, yh = move
        fits_count = 0

        for shape in objectives:
            for xo in range(shape.shape[0]):
                for yo in range(shape.shape[1]):
                    if shape[xo, yo] == 0:
                        continue

                    if xo > xh or yo > yh:
                        continue

                    if (shape.shape[0] - xo > self.board.shape[0] - xh) or \
                       (shape.shape[1] - yo > self.board.shape[1] - yh):
                        continue

                    fits = True
                    for x in range(shape.shape[0]):
                        for y in range(shape.shape[1]):
                            if shape[x, y] == 0:
                                continue

                            pos = (xh - xo + x, yh - yo + y)
                            if shape[x, y] >= 0:
                                if self.board.value(pos) != player:
                                    fits = False
                                    break
                            else:
                                if self.board.value(pos) not in [player, 0]:
                                    fits = False
                                    break

                        if not fits:
                            break

                    if fits:
                        fits_count += 1

        return fits_count

    def get_player_score(self, move, player):
        wins = self.get_fits_count(move, player, self.objectives)
        incomplete_shapes = self.get_incomplete_shapes()
        near_wins = self.get_fits_count(move, player, incomplete_shapes)
        score = wins * self.win_factor + near_wins * self.near_win_factor

        return score

    def get_score(self, move, hero, opponent):
        score = 0
        score += self.get_player_score(move, hero)
        score -= self.get_player_score(move, opponent)

        return score
