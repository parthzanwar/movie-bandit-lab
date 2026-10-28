"""Read-only tools that let the GenAI assistant inspect the existing bandit."""

from functools import lru_cache
import re

import numpy as np

from agents import EpsilonGreedy, LinUCB, PopularityPolicy, RandomPolicy
from environment import (
    BOLLYWOOD_USER_ID,
    GENRE_NAMES,
    MovieBanditEnvironment,
    ORACLE_GENRE_WEIGHT,
    ORACLE_INTERCEPT,
    ORACLE_POPULARITY_WEIGHT,
    candidate_movies,
    demo_user_contexts,
    movie_genre_vectors,
    movie_popularity_score,
    movie_titles,
    train,
    user_contexts,
)


POLICY_NAMES = ["Random", "Popularity", "Epsilon-Greedy", "LinUCB"]
PROFILE_TARGETS = [
    ("Action junkie", ["Action"]),
    ("Rom-com fan", ["Romance", "Comedy"]),
    ("Sci-fi dreamer", ["Sci-Fi"]),
    ("Drama devotee", ["Drama"]),
    ("Comedy fan", ["Comedy"]),
    ("Mystery sleuth", ["Mystery"]),
    ("Horror buff", ["Horror"]),
    ("Adventure seeker", ["Adventure"]),
    ("Animation fan", ["Animation"]),
    ("War-story watcher", ["War"]),
    ("Bollywood buff", None),
]


def build_profile_users():
    """Use the same genre-alignment rule as the existing Streamlit demo."""
    ordered_users = sorted(user_contexts)
    used_users = set()
    selected = {}
    for label, target_genres in PROFILE_TARGETS:
        if target_genres is None:
            selected[label] = BOLLYWOOD_USER_ID
            continue
        target = np.zeros(len(GENRE_NAMES), dtype=float)
        for genre in target_genres:
            target[GENRE_NAMES.index(genre)] = 1.0 / len(target_genres)
        ranked = sorted(
            ordered_users,
            key=lambda user_id: float(user_contexts[user_id] @ target),
            reverse=True,
        )
        selected[label] = next(
            (user_id for user_id in ranked if user_id not in used_users), ranked[0]
        )
        used_users.add(selected[label])
    return selected


PROFILE_USERS = build_profile_users()
candidate_train = train[train.item_id.isin(candidate_movies)]
candidate_train_like_rates = (
    candidate_train.assign(liked=(candidate_train.rating >= 4).astype(int))
    .groupby("item_id").liked.mean()
)
candidate_train_rating_counts = candidate_train.groupby("item_id").size()
popularity_rates = np.array(
    [
        candidate_train_like_rates.get(item_id, movie_popularity_score[item_id])
        for item_id in candidate_movies
    ],
    dtype=float,
)


def _make_policy(name, seed):
    if name == "Random":
        return RandomPolicy(len(candidate_movies), random_state=seed)
    if name == "Popularity":
        return PopularityPolicy(popularity_rates)
    if name == "Epsilon-Greedy":
        return EpsilonGreedy(len(candidate_movies), epsilon=0.1, random_state=seed)
    return LinUCB(len(candidate_movies), context_size=len(GENRE_NAMES), alpha=1.0)


@lru_cache(maxsize=32)
def _run_demo_cached(profile_label: str, n_rounds: int, seed: int = 17):
    """Reproduce the app's seed schedule and per-policy simulation loop."""
    user_id = PROFILE_USERS[profile_label]
    environments = {
        name: MovieBanditEnvironment(random_state=seed + index)
        for index, name in enumerate(POLICY_NAMES)
    }
    policies = {
        name: _make_policy(name, seed + index)
        for index, name in enumerate(POLICY_NAMES)
    }
    oracle = environments["Random"].predict_probabilities(user_id)
    best_probability = float(oracle.max())
    rewards = {name: 0 for name in POLICY_NAMES}
    regrets = {name: 0.0 for name in POLICY_NAMES}
    decisions = {name: [] for name in POLICY_NAMES}

    for round_index in range(n_rounds):
        for name in POLICY_NAMES:
            environment = environments[name]
            policy = policies[name]
            context = environment.reset(user_id)
            means = bonuses = scores = None
            greedy_arm = None
            if name == "LinUCB":
                means = np.empty(policy.n_arms, dtype=float)
                bonuses = np.empty(policy.n_arms, dtype=float)
                for arm in range(policy.n_arms):
                    inverse = np.linalg.inv(policy.A[arm])
                    theta = inverse @ policy.b[arm]
                    means[arm] = theta @ context
                    bonuses[arm] = policy.alpha * np.sqrt(
                        max(0.0, context @ inverse @ context)
                    )
                scores = means + bonuses
                greedy_arm = int(np.argmax(means))

            action = policy.select_action(context)
            reward = environment.step(action)
            if name == "Random":
                behavior = "Exploration"
            elif name == "Popularity":
                behavior = "Exploitation"
            elif name == "Epsilon-Greedy":
                behavior = "Exploration" if policy.last_exploring else "Exploitation"
            else:
                behavior = "Exploration" if action != greedy_arm else "Exploitation"

            item_id = int(candidate_movies[action])
            decisions[name].append(
                {
                    "round": round_index + 1,
                    "movie_id": item_id,
                    "movie": movie_titles[item_id],
                    "oracle_p_like": float(oracle[action]),
                    "reward": int(reward),
                    "behavior": behavior,
                    "linucb_estimate": float(means[action]) if means is not None else None,
                    "ucb_bonus": float(bonuses[action]) if bonuses is not None else None,
                    "ucb_score": float(scores[action]) if scores is not None else None,
                }
            )
            policy.update(action, context, reward)
            rewards[name] += int(reward)
            regrets[name] += best_probability - float(oracle[action])

    return {
        "rewards": rewards,
        "avg_regrets": {name: regrets[name] / n_rounds for name in POLICY_NAMES},
        "decisions": decisions,
    }


def profile_summary(profile_label: str):
    if profile_label not in PROFILE_USERS:
        raise ValueError("Unknown viewer profile.")
    user_id = PROFILE_USERS[profile_label]
    context = MovieBanditEnvironment._context_for_user(user_id)
    top = np.argsort(context)[::-1]
    return {
        "label": profile_label,
        "kind": "synthetic" if user_id == BOLLYWOOD_USER_ID else "MovieLens training user",
        "user_id": int(user_id),
        "top_genres": [
            {"genre": GENRE_NAMES[int(index)], "weight": float(context[index])}
            for index in top
            if context[index] > 0
        ][:5],
    }


def top_movies(profile_label: str, limit: int = 5):
    if profile_label not in PROFILE_USERS:
        raise ValueError("Unknown viewer profile.")
    limit = max(1, min(int(limit), len(candidate_movies)))
    user_id = PROFILE_USERS[profile_label]
    context = MovieBanditEnvironment._context_for_user(user_id)
    probabilities = MovieBanditEnvironment(random_state=0).predict_probabilities(user_id)
    ranked = np.argsort(probabilities)[::-1][:limit]
    movies = []
    for arm in ranked:
        movie_id = int(candidate_movies[int(arm)])
        genre_match = float(context @ movie_genre_vectors[movie_id])
        genres = [
            GENRE_NAMES[index]
            for index, value in enumerate(movie_genre_vectors[movie_id])
            if value > 0
        ]
        movies.append(
            {
                "rank": len(movies) + 1,
                "movie_id": movie_id,
                "title": movie_titles[movie_id],
                "p_like": float(probabilities[int(arm)]),
                "genre_match": genre_match,
                "popularity": float(movie_popularity_score[movie_id]),
                "genres": genres,
            }
        )
    return movies


def _title_variants(title: str):
    """Build simple case-insensitive aliases for matching a movie in a question."""
    base = re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()
    tokens = re.findall(r"[a-z0-9]+", base.casefold())
    variants = set()
    if tokens:
        variants.add(" ".join(tokens))
    if len(tokens) > 1 and tokens[0] in {"the", "a", "an"}:
        variants.add(" ".join(tokens[1:]))
    if len(tokens) > 1 and tokens[-1] in {"the", "a", "an"}:
        variants.add(" ".join(tokens[:-1]))
        variants.add(" ".join([tokens[-1], *tokens[:-1]]))
    return {variant for variant in variants if len(variant) >= 4}


def movie_mentioned_in_query(query: str):
    """Return a canonical candidate title when the question names one."""
    query_tokens = re.findall(r"[a-z0-9]+", query.casefold())
    normalized_query = " " + " ".join(query_tokens) + " "
    matches = []
    for title in set(movie_titles.values()):
        for variant in _title_variants(title):
            if " " + variant + " " in normalized_query:
                matches.append((len(variant), title))
    if not matches:
        return None
    # Prefer the longest title phrase, which avoids matching a short title
    # when a longer candidate name is also present in the same question.
    return max(matches, key=lambda item: item[0])[1]


def inspect_movie_for_profile(profile_label: str, requested_title: str):
    """Return the simulator's exact fit, probability, and rank for one movie."""
    if profile_label not in PROFILE_USERS:
        raise ValueError("Unknown profile.")
    title_key = next(
        (
            item_title
            for item_title in set(movie_titles.values())
            if item_title.casefold() == requested_title.casefold()
            or _title_variants(item_title) & _title_variants(requested_title)
        ),
        None,
    )
    if title_key is None:
        raise ValueError("That movie is not in the candidate slate.")
    movie_id = next(item_id for item_id, title in movie_titles.items() if title == title_key)
    if movie_id not in candidate_movies:
        raise ValueError("That movie is not in the candidate slate.")

    user_id = PROFILE_USERS[profile_label]
    context = MovieBanditEnvironment._context_for_user(user_id)
    probabilities = MovieBanditEnvironment(random_state=0).predict_probabilities(user_id)
    arm_index = list(candidate_movies).index(movie_id)
    probability = float(probabilities[arm_index])
    order = np.argsort(probabilities)[::-1]
    rank = int(np.flatnonzero(order == arm_index)[0]) + 1
    genre_match = float(context @ movie_genre_vectors[movie_id])
    popularity = float(movie_popularity_score[movie_id])
    genre_term = float(ORACLE_GENRE_WEIGHT * genre_match)
    popularity_term = float(ORACLE_POPULARITY_WEIGHT * popularity)
    logit = float(genre_term + popularity_term + ORACLE_INTERCEPT)
    genres = [
        GENRE_NAMES[index]
        for index, value in enumerate(movie_genre_vectors[movie_id])
        if value > 0
    ]
    is_demo = movie_id < 0
    training_rating_count = int(candidate_train_rating_counts.get(movie_id, 0))
    training_like_rate = candidate_train_like_rates.get(movie_id)
    profile = profile_summary(profile_label)

    return {
        "profile_label": profile_label,
        "profile_kind": profile["kind"],
        "user_id": profile["user_id"],
        "profile_top_genres": profile["top_genres"][:5],
        "movie_id": int(movie_id),
        "movie_title": title_key,
        "movie_genres": genres,
        "movie_source": "Bollywood demo" if is_demo else "MovieLens 100K",
        "movie_rank": rank,
        "candidate_count": len(candidate_movies),
        "genre_match": genre_match,
        "genre_term": genre_term,
        "popularity": popularity,
        "popularity_term": popularity_term,
        "popularity_basis": (
            "hand-assigned illustrative demo value; this item has no MovieLens ratings"
            if is_demo
            else "scaled high-rating counts from MovieLens training data"
        ),
        "training_rating_count": training_rating_count,
        "training_like_rate": (
            float(training_like_rate) if training_like_rate is not None else None
        ),
        "oracle_logit": logit,
        "oracle_p_like": probability,
        "top_movies": top_movies(profile_label, limit=5),
    }


def compare_policies(profile_label: str, rounds: int):
    if profile_label not in PROFILE_USERS:
        raise ValueError("Unknown viewer profile.")
    if not 10 <= int(rounds) <= 500:
        raise ValueError("Rounds must be between 10 and 500.")
    result = _run_demo_cached(profile_label, int(rounds), 17)
    return [
        {
            "policy": name,
            "likes": int(result["rewards"][name]),
            "like_rate": float(result["rewards"][name] / int(rounds)),
            "average_regret": float(result["avg_regrets"][name]),
        }
        for name in POLICY_NAMES
    ]


def round_decision(profile_label: str, policy: str, rounds: int, round_number: int):
    if policy not in POLICY_NAMES:
        raise ValueError("Unknown policy.")
    if not 10 <= int(rounds) <= 500:
        raise ValueError("Rounds must be between 10 and 500.")
    if not 1 <= int(round_number) <= int(rounds):
        raise ValueError("Round number must be inside the selected horizon.")
    result = _run_demo_cached(profile_label, int(rounds), 17)
    return result["decisions"][policy][int(round_number) - 1]

