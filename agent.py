import numpy as np
import sys
import itertools
from collections import Counter
import math

class Agent:
    def __init__(self, epsilon, delta):
        self.epsilon = epsilon
        self.delta = delta
        self.model = None
        self.prev_interp = None
        self.prev_pred = None

    def compute_required_training_dataset_size(self):
        lit_len = len(self.model)
        e = self.epsilon
        d = self.delta

        m = lit_len / e * math.log(lit_len / d)
        m = int(m+1)

        return m

    def process_first_observation(self, interpretation):
        # TODO: you can't check reward here. Anything else?
        self.prev_interp = interpretation
        return False

    def predict(self, interpretation, reward):
        prediction = None

        if reward is not None:

            if reward is 0: # Only change model if guess was wrong
                model = []
                for version in self.model:
                    if self.check_sat(version, self.prev_interp):
                        model.append(version)

                self.model = model

            prediction = False # TODO: Always predict False in training phase? If remove, need to check the update loop
        else:
            prediction = True
            for version in self.model:
                if self.check_sat(version, interpretation) == False:
                    prediction = False
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
        n = self.n_variables
        j = self.j

        perm = list(itertools.product([True, False ,None], repeat=n))

        allowed_clauses = []
        for clause in perm:
            nbr_None = clause.count(None)
            if nbr_None >= n - j and nbr_None < n:
                allowed_clauses.append(clause)

        self.model = allowed_clauses


    def check_sat(self, version, inter):
        for i, var in enumerate(inter):
            if var == version[i]:
                return True

        return False


def pause():
    programPause = input("Press the <ENTER> key to continue...")
