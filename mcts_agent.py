# Max van der Ree (6241638)

from types import SimpleNamespace
import random, copy, time
import numpy as np
import math

import game as g
from random_agent import RandomAgent


class Agent:
    def __init__(self, iterations, id, verbosity="info", branch=None, c=None):
        self.iterations = iterations
        self.id = id
        self.strategy = MCTS(iterations, verbosity, branch, c)

    def make_move(self, game):
        move = self.strategy.get_move(game)

        return move

    def __str__(self):
        return f'Player {self.id} (MCTS Agent)'


class MCTS:

    def __init__(self, iterations, verbosity, branch, c):
        self.iterations = iterations
        self.verbosity = verbosity
        self.branch = branch
        self.c = c

    def get_move(self, game):
        self.tree = Tree(self.verbosity, game, self.c, self.branch)
        self.tree.grow(iterations=self.iterations)
        choice = self.tree.root.select_child(iteration=self.iterations)
        if self.verbosity == "debug":
            print(f"Choice: {choice}")

        return choice.move


class Tree:

    max_branch_factor = 50

    def __init__(self, verbosity, game, c, branch):
        self.verbosity = verbosity
        if branch is not None:
            self.max_branch_factor = branch

        self.root = Node(
            game,
            player=game.players[0],  # We assume the first player is the MCTS Agent
            opponent=game.players[1],
            depth=0,
            verbosity=self.verbosity,
            state=copy.deepcopy(game.board),
            type_factor=1,
            c=c,
        )

    def backpropagate(self, node, reward):
        node.update(reward)
        if node.parent:
            self.backpropagate(node.parent, reward)

    def select(self, node, iteration):
        child = node.select_child(iteration)
        if child.children:
            return self.select(child, iteration)
        else:
            return child

    def grow(self, iterations):
        for iteration in range(1, iterations+1):
            node = self.select(self.root, iteration)
            reward = node.simulate()
            self.backpropagate(node.parent, reward)

        if self.verbosity == "debug":
            print("tree:\n", self, "\n")

    def get_child_hierarchy(self, node):
        hierarchy = {}
        for child in node.children:
            if child.children:
                hierarchy[str(child)] = self.get_child_hierarchy(child)
            else:
                hierarchy[str(child)] = None

        return hierarchy

    def __str__(self):
        tree = {}
        tree[str(self.root)] = self.get_child_hierarchy(self.root)
        import json
        return json.dumps(tree, indent=4)


class Node:

    max_branch_factor = 92
    mock_visits = 0.1
    explored = True

    def __init__(
        self,
        game,
        player=None,
        opponent=None,
        state=None,
        parent=None,
        move=None,
        verbosity=None,
        depth=None,
        type_factor=1,
        c=None,
    ):
        self.opponent = opponent
        self.player = player
        self.parent = parent
        self.game = game
        self.state = state
        self.id = random.randint(0, 1000000)
        self.rewards_sum = 0
        self.reward = 0
        self.visits = 0
        self.c = c
        self.depth = depth
        self.type_factor = type_factor
        self.selector = UCB(verbosity, type_factor, c)
        self.children = []
        self.unexplored_options = self.init_unexplored_options()
        self.move = move

    def init_unexplored_options(self):
        unexplored_options = []
        unexplored_moves = [tuple(pos) for pos in self.state.free_positions()]
        for _ in range(self.max_branch_factor):
            if not unexplored_moves:
                break

            move = random.choice(unexplored_moves)
            type_factor = 1 - self.type_factor  # Child's type factor is inverse of parent's
            node = SimpleNamespace(
                reward=0, move=move, visits=self.mock_visits, type_factor=type_factor, explored=False
            )
            unexplored_options.append(node)
            unexplored_moves.remove(move)

        return unexplored_options

    def expand(self):
        unexplored_option = random.choice(self.unexplored_options)
        move = unexplored_option.move
        child_state = copy.deepcopy(self.state)
        child_state.place(move, self.player.id)
        child = Node(
            self.game,
            player=self.opponent,  # Switch player and opponent
            opponent=self.player,
            state=child_state,
            parent=self,
            move=move,
            c=self.c,
            depth=self.depth + 1,
            type_factor=1-self.type_factor  # Child's type factor is inverse of parent's
        )
        self.children.append(child)
        self.unexplored_options.remove(unexplored_option)

        return child

    def simulate(self):
        sim = Simulation(self.state, self.move, self.game.objectives)
        reward = sim.rollout()
        self.update(reward)

        return reward

    def update(self, reward):
        self.visits += 1
        self.rewards_sum += reward
        self.reward = self.rewards_sum / self.visits

    def select_child(self, iteration=None):
        selection = self.selector.select(self.children + self.unexplored_options, iteration=iteration)
        if selection is None:  # If selection is None, this means the selector has opted to explore
            selection = self.expand()

        return selection

    def __str__(self):
        return f"{self.id} -- reward:{self.reward} | visits:{self.visits}"


class UCB:

    c = 0.4

    def __init__(self, verbosity, type_factor, c):
        self.verbosity = verbosity
        if c is not None:
            self.c = c

        self.type_factor = type_factor

    def get_confidence_bound(self, reward, visits, iteration):
        reward = self.type_factor * reward  # Negative type factor will invert reward, in case parent node is minimizer
        return reward + self.c * math.sqrt(math.log(iteration) / visits)

    def get_confidence_bounds(self, child_nodes, iteration):
        confidence_bounds = []
        for child in child_nodes:
            bound = self.get_confidence_bound(child.reward, child.visits, iteration)
            confidence_bounds.append((child, bound))

        confidence_bounds.sort(key=lambda x: x[1])
        return confidence_bounds

    def select(self, child_nodes, iteration):
        confidence_bounds = self.get_confidence_bounds(child_nodes, iteration)
        selection = confidence_bounds[-1][0]
        if not selection.explored:
            return None

        return selection


class Simulation:

    hero = 1
    opponent = 2

    def __init__(self, board, move, objectives):
        self.move = move
        self.players = [RandomAgent(self.opponent), RandomAgent(self.hero)]  # the order is important (opponent should do the next move)
        self.board = copy.deepcopy(board)
        self.board.place(self.move, self.hero)
        self.game = g.Game.from_board(
            self.board, objectives, self.players, False
        )

    def normalize_result(self, result):
        return result * 2 - 1

    def rollout(self):
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

        result = self.normalize_result(result)

        return result
