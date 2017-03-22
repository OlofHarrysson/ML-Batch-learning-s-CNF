import numpy as np

class Agent:
    def __init__(self, epsilon, delta):
        self.epsilon = epsilon
        self.delta = delta

    def compute_required_training_dataset_size(self):
        # TODO - compute your value here.
        # It should depend on self.n_variables and self.j, self.epsilon and self.delta.
        return 100

    def process_first_observation(self, interpretation):
        # TODO - do something with interpretation and return
        # a prediction
        return False

    def predict(self, interpretation, reward):
        if reward is not None:
            ...
            # We are in training branch
            #
            # Use the reward and the previous interpretation and
            # the previous prediction to update your model.
            # Then make a prediction for the given interpretation.
        else:
            ...
            # We are in testing branch
            # Only make a prediction.

        return False # TODO - return your prediction

    def interact_with_oracle(self, oracle_session):
        # You may alter this method as you desire,
        # but it is not required.

        self.n_variables, self.j = oracle_session.request_parameters()

        m = self.compute_required_training_dataset_size()
        first_sample = oracle_session.request_dataset(m)
        prediction = self.process_first_observation(first_sample)

        while oracle_session.has_more_samples():
            interpretation, reward = oracle_session.predict(prediction)
            prediction = self.predict(interpretation, reward)


