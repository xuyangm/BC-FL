import random


class Selector(object):
    """
    Select participants in each round.
    """

    def __init__(self, candidates):
        self.candidates = candidates

    def select_participants(self, sample_size):
        participants = random.sample(self.candidates, sample_size)
        return participants
