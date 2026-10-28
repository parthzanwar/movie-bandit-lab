"""Simple numpy-only policies for the movie recommendation bandit."""

import numpy as np


# This policy explores without using past rewards. It is a useful baseline for
# seeing how well blind exploration does when every movie is equally likely.
class RandomPolicy:
    def __init__(self, n_arms, random_state=None):
        self.n_arms = n_arms
        self.rng = np.random.default_rng(random_state)

    def select_action(self, context):
        return int(self.rng.integers(self.n_arms))

    def update(self, action, context, reward):
        pass


# This policy always chooses the movie with the highest training-set like
# rate. It never changes its choice based on users or rewards during a run.
class PopularityPolicy:
    def __init__(self, train_like_rates):
        self.train_like_rates = np.asarray(train_like_rates, dtype=float)

    def select_action(self, context):
        return int(np.argmax(self.train_like_rates))

    def update(self, action, context, reward):
        pass


# This is a plain multi-armed bandit: it tracks one average reward per movie
# and ignores the user's context. Epsilon makes it try other movies sometimes.
class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1, random_state=None):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.rng = np.random.default_rng(random_state)
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms, dtype=float)
        # Keep this flag so the demo can explain whether this exact choice
        # came from the epsilon exploration branch or the current best arm.
        self.last_exploring = False

    def select_action(self, context):
        self.last_exploring = self.rng.random() < self.epsilon
        if self.last_exploring:
            return int(self.rng.integers(self.n_arms))
        return int(np.argmax(self.values))

    def update(self, action, context, reward):
        self.counts[action] += 1
        # The sample average gives each observed reward the right weight.
        self.values[action] += (reward - self.values[action]) / self.counts[action]


# LinUCB learns a separate linear reward model for each movie. The upper
# confidence bonus encourages trying movies where the current user is novel.
class LinUCB:
    def __init__(self, n_arms, context_size, alpha=1.0):
        self.n_arms = n_arms
        self.context_size = context_size
        self.alpha = alpha
        # Starting with identity matrices is like a small ridge prior for each arm.
        self.A = np.repeat(np.eye(context_size)[None, :, :], n_arms, axis=0)
        self.b = np.zeros((n_arms, context_size), dtype=float)

    def select_action(self, context):
        x = np.asarray(context, dtype=float)
        scores = np.empty(self.n_arms, dtype=float)
        for arm in range(self.n_arms):
            inverse = np.linalg.inv(self.A[arm])
            theta = inverse @ self.b[arm]
            mean_reward = theta @ x
            uncertainty = np.sqrt(max(0.0, x @ inverse @ x))
            scores[arm] = mean_reward + self.alpha * uncertainty
        return int(np.argmax(scores))

    def update(self, action, context, reward):
        x = np.asarray(context, dtype=float)
        # Add this user's context to the chosen movie's ridge regression only.
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x
