import random
from collections import OrderedDict


class Selector(object):
    """
    Select participants in each round.
    """

    def __init__(self, candidates: list):
        self.explored = OrderedDict()
        self.unexplored = candidates
        self.explore_ratio = 0.9
        self.candidates_size = len(candidates)

    def select_participants(self, sample_size, method='Random'):
        participants = []
        if method == 'Random':
            participants = random.sample(self.unexplored, sample_size)

        elif method == 'Bandit':
            explore_num = min(len(self.unexplored), int(self.explore_ratio * sample_size))
            exploit_num = sample_size - explore_num

            if explore_num == 0:
                print("stop exploration now, reset")
                self.unexplored = list(self.explored.keys())
                self.explored = OrderedDict()
                return self.select_participants(sample_size, method)

            if len(self.explored) < exploit_num:
                participants = random.sample(self.unexplored, sample_size)
            else:
                unexplored_participants = random.sample(self.unexplored, explore_num)
                explored_participants = sorted(self.explored, key=self.explored.get, reverse=True)[:exploit_num]
                participants = unexplored_participants + explored_participants

        return participants

    def update_contribution(self, shapley_values):
        print(shapley_values)
        for candidate in shapley_values:
            if candidate in self.unexplored:
                self.unexplored.remove(candidate)
            self.explored[candidate] = shapley_values[candidate]
