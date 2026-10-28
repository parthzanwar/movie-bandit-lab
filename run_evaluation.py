"""Compare four movie bandit policies on repeatable user sequences."""

import matplotlib.pyplot as plt
import numpy as np

from agents import EpsilonGreedy, LinUCB, PopularityPolicy, RandomPolicy
from environment import (
    BOLLYWOOD_ITEM_IDS,
    MovieBanditEnvironment,
    candidate_movies,
    movie_titles,
    movie_popularity_score,
    train,
    user_contexts,
)


# These settings make each run long enough to learn, while keeping the example
# fast to repeat. Each seed gets its own user and reward random streams.
N_SEEDS = 5
N_ROUNDS = 2000
N_ARMS = len(candidate_movies)
user_ids = np.array(sorted(user_contexts), dtype=int)
bollywood_item_id_set = set(BOLLYWOOD_ITEM_IDS)

print(f"Candidate slate: {N_ARMS} movies")
print(f"Evaluation seeds: 0 through {N_SEEDS - 1}; rounds per policy and seed: {N_ROUNDS}")
for slate_index, item_id in enumerate(candidate_movies):
    print(f"  {slate_index:02d}. {movie_titles[item_id]} [item_id={item_id}]")


# Calculate each candidate's like rate from training ratings only. The fixed
# popularity policy uses this once and does not adapt while the experiment runs.
candidate_train = train[train.item_id.isin(candidate_movies)]
candidate_stats = candidate_train.assign(
    liked=(candidate_train.rating >= 4).astype(int)
).groupby("item_id").liked.mean()
train_like_rates = np.array(
    [candidate_stats.get(item_id, movie_popularity_score[item_id]) for item_id in candidate_movies]
)


# Cache oracle probabilities for every user and candidate once. They remain
# fixed during learning, and this avoids recalculating the same predictions
# thousands of times. Rows follow user_ids; columns follow candidate_movies.
probability_env = MovieBanditEnvironment(random_state=0)
oracle_probabilities = np.vstack(
    [probability_env.predict_probabilities(user_id) for user_id in user_ids]
)
user_row = {int(user_id): row for row, user_id in enumerate(user_ids)}
best_probabilities = oracle_probabilities.max(axis=1)


# Before learning, check whether the fixed oracle actually gives different
# users different best movies and whether its best options beat typical ones.
diagnostic_rng = np.random.default_rng(42)
diagnostic_users = diagnostic_rng.choice(user_ids, size=50, replace=False)
diagnostic_rows = np.array([user_row[int(user_id)] for user_id in diagnostic_users])
diagnostic_probabilities = oracle_probabilities[diagnostic_rows]
diagnostic_best = diagnostic_probabilities.argmax(axis=1)
diagnostic_gap = (
    diagnostic_probabilities.max(axis=1)
    - np.median(diagnostic_probabilities, axis=1)
)
print(
    "Oracle diagnostic (50 random users): "
    f"{len(np.unique(diagnostic_best))} distinct best-candidate movies; "
    f"mean best-vs-median p(like) gap = {diagnostic_gap.mean():.3f}"
)


# These constructors give each policy a fresh state for each seed. Random
# policies get a seeded generator so the same experiment can be reproduced.
policy_names = ["Random", "Popularity", "Epsilon-Greedy", "LinUCB"]


def make_policies(seed):
    return {
        "Random": RandomPolicy(N_ARMS, random_state=seed),
        "Popularity": PopularityPolicy(train_like_rates),
        "Epsilon-Greedy": EpsilonGreedy(
            N_ARMS, epsilon=0.1, random_state=seed
        ),
        "LinUCB": LinUCB(N_ARMS, context_size=len(next(iter(user_contexts.values()))), alpha=1.0),
    }


# For every seed the policies see the same sequence of users. Each policy gets
# its own environment reward stream, and it receives one sampled reward per turn.
reward_runs = {name: [] for name in policy_names}
regret_runs = {name: [] for name in policy_names}
total_rewards = {name: [] for name in policy_names}
bollywood_action_counts = {name: [] for name in policy_names}

for seed in range(N_SEEDS):
    user_rng = np.random.default_rng(seed)
    round_users = user_rng.choice(user_ids, size=N_ROUNDS, replace=True)

    for name, policy in make_policies(seed).items():
        env = MovieBanditEnvironment(random_state=seed + 1000)
        cumulative_reward = np.zeros(N_ROUNDS, dtype=float)
        cumulative_regret = np.zeros(N_ROUNDS, dtype=float)
        reward_sum = 0.0
        regret_sum = 0.0
        bollywood_actions = 0

        for round_index, user_id in enumerate(round_users):
            context = env.reset(int(user_id))
            action = policy.select_action(context)
            if candidate_movies[action] in bollywood_item_id_set:
                bollywood_actions += 1
            reward = env.step(action)
            policy.update(action, context, reward)

            # Regret compares the oracle's best probability for this user to
            # the chosen movie's probability, even though reward is sampled.
            probabilities = oracle_probabilities[user_row[int(user_id)]]
            regret_sum += best_probabilities[user_row[int(user_id)]] - probabilities[action]
            reward_sum += reward
            cumulative_reward[round_index] = reward_sum
            cumulative_regret[round_index] = regret_sum

        reward_runs[name].append(cumulative_reward)
        regret_runs[name].append(cumulative_regret)
        total_rewards[name].append(reward_sum)
        bollywood_action_counts[name].append(bollywood_actions)


# Plot the average learning curves and shade one standard deviation across the
# five seeds. The saved pictures make it easy to compare rewards and regret.
round_axis = np.arange(1, N_ROUNDS + 1)


def plot_curves(runs, title, y_label, output_path):
    plt.figure(figsize=(10, 6))
    for name in policy_names:
        values = np.asarray(runs[name])
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        plt.plot(round_axis, mean, label=name)
        plt.fill_between(round_axis, mean - std, mean + std, alpha=0.18)
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


plot_curves(
    reward_runs,
    "Cumulative reward by policy",
    "Cumulative reward",
    "results_reward.png",
)
plot_curves(
    regret_runs,
    "Cumulative model-relative pseudo-regret by policy",
    "Cumulative pseudo-regret",
    "results_regret.png",
)


# Report each policy's final total reward as mean plus or minus standard
# deviation across seeds, matching the uncertainty shown in the figures.
print(f"Total reward over {N_ROUNDS} rounds (mean +/- std across {N_SEEDS} seeds)")
print(f"{'Policy':<18} {'Mean total reward':>18} {'Std':>10}")
for name in policy_names:
    values = np.asarray(total_rewards[name], dtype=float)
    print(f"{name:<18} {values.mean():>18.1f} {values.std():>10.1f}")
print("\nBollywood actions per seed (out of 2,000 rounds):")
for name in policy_names:
    print(f"{name:<18} {bollywood_action_counts[name]} (total={sum(bollywood_action_counts[name])})")
print("\nSaved plots: results_reward.png and results_regret.png")
