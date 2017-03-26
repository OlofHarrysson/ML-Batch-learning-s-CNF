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
        self.oracle = None # TODO REMOVE

    def compute_required_training_dataset_size(self):
        # TODO - compute your value here.
        # It should depend on self.n_variables and self.j, self.epsilon and self.delta.


        # TODO - How does it depend on j?
        # TODO - n in theorem 3.4 means number of versions right?
        # n = self.n_variables
        lit_len = len(self.model)
        e = self.epsilon
        d = self.delta

        m = lit_len / e * math.log(lit_len / d) # 2n = model length
        # print(m)


        k = 1 / e * math.log(lit_len / d)
        # print(k)

        k = int(k+1)


        # sys.exit(1)
        # m = int(m) + 1


        # m = 1 / e * math.log(2 / d) # 2n unitsteps = 1? Tagen från http://www.cs.princeton.edu/courses/archive/spr06/cos511/scribe_notes/0214.pdf sida 2
        # m = int(m) + 1


        # print(m)
        # print(len(self.model))
        # sys.exit(1)
        # m = 70
        return k

    def process_first_observation(self, interpretation):
        # TODO: you can't check reward here. Anything else?
        self.prev_interp = interpretation
        return False

    def predict(self, interpretation, reward):
        prediction = None

        if reward is not None:

            if reward is 0: # Only change model if guess was wrong
                # print("------------------")
                # print(len(self.model))
                model = []
                for version in self.model:
                    if self.check_sat(version, self.prev_interp):
                        model.append(version)

                # print(len(model))
                # print(model)
                self.model = model
                # pause()


            prediction = False # TODO: Always predict False in training phase?
        else:
            prediction = True
            for version in self.model:
                if self.check_sat(version, interpretation) == False:
                    prediction = False
                    break

        return prediction

    def interact_with_oracle(self, oracle_session):
        self.oracle = oracle_session # TODO REMOVE

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


# Manually debug while loop
# print(reward)
# print("Loop ------")
# print(interpretation)
# print(oracle_session.getcnf())
# pause()

    def init_model(self):
        # permutation = list(itertools.product([True, False], repeat=self.n_variables))
        # model = []
        # for version in permutation:
        #     model.append((version, False))
        #     model.append((version, True))

        # self.model = model

        n = 2
        j = 1

        n = self.n_variables
        j = self.j


        perm = list(itertools.product([True, False ,None], repeat=n))
        # perm = list(itertools.product([True, False, None, None, None], repeat=j)) # 5^3 = 125 -> j=repeat
        # perm = list(itertools.permutations([0,1,None], 3))

        copy = []
        for clause in perm:
            nbr_None = clause.count(None)
            if nbr_None >= n - j and nbr_None < n:
                copy.append(clause)

        # print(self.oracle.getcnf())
        # print(perm2)
        # print(copy)
        # print(len(copy))
        # print(perm)
        # print(len(perm))


        # print("2*n^j = " + str((2*n) ** j))
        # print("n^j = " + str(n ** j))
        self.model = copy

    def check_sat(self, version, inter):
        # print(version)
        # print(list(inter))
        # pause()
        for i, var in enumerate(inter):
            # print(type(inter[i]))
            # pause()
            if var == version[i]:
                return True

        return False



def pause():
    programPause = input("Press the <ENTER> key to continue...")
