import numpy as np
import sys
import itertools
from collections import Counter

class Agent:
    def __init__(self, epsilon, delta):
        self.epsilon = epsilon
        self.delta = delta
        self.model = None
        self.prev_interp = None
        self.prev_pred = None

    def compute_required_training_dataset_size(self):
        # TODO - compute your value here.
        # It should depend on self.n_variables and self.j, self.epsilon and self.delta.
        return 100

    def process_first_observation(self, interpretation):
        # TODO: you can't check reward here. Anything else?
        self.prev_interp = interpretation
        return False

    def predict(self, interpretation, reward):
        prediction = None

        if reward is not None:

            if reward is 0: # Only change model if guess was wrong

                for version in self.model[:]: # Iterate over a copy
                    if version[0] == tuple(self.prev_interp) and version[1] == self.prev_pred:
                        self.model.remove(version)
                        break


        else: # TODO: Need this else?
            pass

        for version in self.model:
            if version[0] == tuple(interpretation):
                prediction = version[1]
                break

        return prediction

    def interact_with_oracle(self, oracle_session):
        self.n_variables, self.j = oracle_session.request_parameters() # X1
        self.init_model()

        m = self.compute_required_training_dataset_size() # Y1
        first_sample = oracle_session.request_dataset(m) # X2

        prediction = self.process_first_observation(first_sample) # Y2
        self.prev_pred = prediction

        while oracle_session.has_more_samples():
            interpretation, reward = oracle_session.predict(prediction) # X3
            prediction = self.predict(interpretation, reward) # Y3
            self.prev_pred = prediction
            self.prev_interp = interpretation


    def init_model(self):
        permutation = list(itertools.product([True, False], repeat=self.n_variables))
        model = []
        for version in permutation:
            model.append((version, False))
            model.append((version, True))

        self.model = model


def pause():
    programPause = input("Press the <ENTER> key to continue...")



