"""A small contextual bandit environment built from MovieLens 100K."""

from pathlib import Path

import numpy as np
import pandas as pd


# These names follow the order used in MovieLens u.item, so every genre flag
# lines up with the same position in a user's context and a movie's features.
GENRE_NAMES = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]
RATING_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
MOVIE_COLUMNS = [
    "item_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    *GENRE_NAMES,
]
DATA_DIR = Path(__file__).parent / "data" / "ml-100k"


# Read the two source files once. We keep the movie table because its genre
# flags are the common language used to describe both users and movies.
ratings = pd.read_csv(DATA_DIR / "u.data", sep="\t", names=RATING_COLUMNS)
movies = pd.read_csv(
    DATA_DIR / "u.item", sep="|", names=MOVIE_COLUMNS, encoding="latin-1"
)
GENRE_MATRIX = movies[GENRE_NAMES].to_numpy(dtype=float)
MOVIE_ROW = {int(movie_id): row for row, movie_id in enumerate(movies.item_id)}


# Sort each person's history by time, then take their first four fifths for
# training and last fifth for testing. The oracle only sees the training part.
ordered = ratings.sort_values(["user_id", "timestamp"], kind="mergesort")
train_parts = []
test_parts = []
for _, user_ratings in ordered.groupby("user_id", sort=False):
    split_at = int(len(user_ratings) * 0.8)
    train_parts.append(user_ratings.iloc[:split_at])
    test_parts.append(user_ratings.iloc[split_at:])
train = pd.concat(train_parts, ignore_index=True)
test = pd.concat(test_parts, ignore_index=True)


# Count training ratings and keep only movies with enough examples. To spread
# the fixed 30 choices across genres, we repeatedly pick the movie that adds
# the most genre coverage, using popularity to break ties.
training_counts = train.groupby("item_id").size().to_dict()
eligible = [
    int(item_id)
    for item_id, count in training_counts.items()
    if count >= 50 and int(item_id) in MOVIE_ROW
]
eligible.sort(key=lambda item_id: (-training_counts[item_id], item_id))
candidate_movies = []
covered_genres = np.zeros(len(GENRE_NAMES), dtype=bool)
remaining = set(eligible)
while remaining and len(candidate_movies) < 30:
    def candidate_score(item_id):
        genre_flags = GENRE_MATRIX[MOVIE_ROW[item_id]] > 0
        new_genres = np.count_nonzero(genre_flags & ~covered_genres)
        return (new_genres, training_counts[item_id], -item_id)

    chosen = max(remaining, key=candidate_score)
    candidate_movies.append(chosen)
    covered_genres |= GENRE_MATRIX[MOVIE_ROW[chosen]] > 0
    remaining.remove(chosen)

if len(candidate_movies) != 30:
    raise ValueError(
        f"Only found {len(candidate_movies)} movies with at least 50 training ratings; need 30."
    )


# Keep the original training-selected slate intact, then append clearly marked
# hand-added demo films. Negative IDs cannot collide with MovieLens item IDs.
movielens_candidate_movies = tuple(candidate_movies)
BOLLYWOOD_DEMO_ITEMS = {
    -1: {
        "title": "3 Idiots (2009)",
        "genres": ["Comedy", "Drama", "Romance"],
        "popularity": 0.98,
    },
    -2: {
        "title": "Dangal (2016)",
        "genres": ["Drama"],
        "popularity": 0.96,
    },
    -3: {
        "title": "Sholay (1975)",
        "genres": ["Action", "Adventure", "Comedy", "Drama"],
        "popularity": 0.94,
    },
    -4: {
        "title": "Dil Chahta Hai (2001)",
        "genres": ["Comedy", "Drama", "Romance"],
        "popularity": 0.88,
    },
    -5: {
        "title": "Gully Boy (2019)",
        "genres": ["Drama", "Musical"],
        "popularity": 0.84,
    },
    -6: {
        "title": "Queen (2013)",
        "genres": ["Comedy", "Drama", "Romance"],
        "popularity": 0.86,
    },
    -7: {
        "title": "Andhadhun (2018)",
        "genres": ["Comedy", "Crime", "Thriller"],
        "popularity": 0.85,
    },
    -8: {
        "title": "Lagaan (2001)",
        "genres": ["Drama", "Musical", "Romance"],
        "popularity": 0.91,
    },
    -9: {
        "title": "PK (2014)",
        "genres": ["Comedy", "Drama", "Sci-Fi"],
        "popularity": 0.95,
    },
    -10: {
        "title": "Zindagi Na Milegi Dobara (2011)",
        "genres": ["Adventure", "Comedy", "Drama"],
        "popularity": 0.90,
    },
    -11: {
        "title": "Chak De! India (2007)",
        "genres": ["Drama"],
        "popularity": 0.89,
    },
    -12: {
        "title": "Barfi! (2012)",
        "genres": ["Comedy", "Drama", "Romance"],
        "popularity": 0.87,
    },
}
BOLLYWOOD_ITEM_IDS = tuple(BOLLYWOOD_DEMO_ITEMS)
candidate_movies = movielens_candidate_movies + BOLLYWOOD_ITEM_IDS

# Convert the hand-written genre labels into the same 19-position flag vectors
# used by MovieLens, so the bandit sees one consistent kind of movie context.
movie_genre_vectors = {
    int(item_id): GENRE_MATRIX[MOVIE_ROW[int(item_id)]].copy()
    for item_id in movies.item_id
}
movie_titles = {int(row.item_id): row.title for row in movies.itertuples()}
for item_id, details in BOLLYWOOD_DEMO_ITEMS.items():
    vector = np.zeros(len(GENRE_NAMES), dtype=float)
    for genre in details["genres"]:
        vector[GENRE_NAMES.index(genre)] = 1.0
    movie_genre_vectors[item_id] = vector
    movie_titles[item_id] = details["title"]


# Build a taste vector from movies rated four or five by each user. Dividing by
# the total genre count makes the values add up to one when the user has likes.
user_contexts = {}
for user_id, user_ratings in train.groupby("user_id"):
    liked_ids = user_ratings.loc[user_ratings.rating >= 4, "item_id"]
    genre_counts = np.zeros(len(GENRE_NAMES), dtype=float)
    for item_id in liked_ids:
        row = MOVIE_ROW.get(int(item_id))
        if row is not None:
            genre_counts += GENRE_MATRIX[row]
    total = genre_counts.sum()
    if total > 0:
        genre_counts /= total
    user_contexts[int(user_id)] = genre_counts


# Add one explicitly synthetic profile for the demo picker. Averaging its genre
# flags gives it a broad taste matching the hand-added Bollywood slate; it is
# kept separate so evaluation still samples only real training users.
BOLLYWOOD_USER_ID = 0
bollywood_profile = np.sum(
    [movie_genre_vectors[item_id] for item_id in BOLLYWOOD_ITEM_IDS], axis=0
)
if bollywood_profile.sum() > 0:
    bollywood_profile /= bollywood_profile.sum()
demo_user_contexts = {BOLLYWOOD_USER_ID: bollywood_profile}


# Count high ratings per movie using only the training split, then scale the
# counts to [0, 1] across the fixed candidates. These frozen scores summarize
# which movies were liked often in TRAIN without looking at future feedback.
training_like_counts = (
    train.loc[train.rating >= 4].groupby("item_id").size().to_dict()
)
candidate_like_counts = np.array(
    [training_like_counts.get(item_id, 0) for item_id in movielens_candidate_movies],
    dtype=float,
)
like_min = float(candidate_like_counts.min())
like_max = float(candidate_like_counts.max())
if like_max == like_min:
    candidate_popularity = np.zeros(len(candidate_movies), dtype=float)
else:
    candidate_popularity = (candidate_like_counts - like_min) / (like_max - like_min)
movie_popularity_score = {
    item_id: float(candidate_popularity[index])
    for index, item_id in enumerate(movielens_candidate_movies)
}
# These hand-assigned demo scores share the same [0, 1] scale as the normalized
# MovieLens scores above; they are illustrative, not measured MovieLens data.
movie_popularity_score.update(
    {
        item_id: float(details["popularity"])
        for item_id, details in BOLLYWOOD_DEMO_ITEMS.items()
    }
)


# Personalization needs a user-specific preference signal: the same movie can
# suit one person's liked genres and miss another's. We freeze this simple
# train-derived response rule so learning agents face a stable oracle. Its
# genre term rewards overlap with that user's training likes; popularity gives
# every user the same small bonus for movies liked often in TRAIN.
ORACLE_GENRE_WEIGHT = 4.0
ORACLE_POPULARITY_WEIGHT = 1.0
ORACLE_INTERCEPT = -1.5


# The environment holds the current user and returns their fixed taste vector.
# Choosing an action means choosing a zero-based position in candidate_movies.
class MovieBanditEnvironment:
    def __init__(self, random_state=None):
        self.rng = np.random.default_rng(random_state)
        self.user_id = None
        self.context = None

    def reset(self, user_id):
        user_id = int(user_id)
        if user_id not in user_contexts and user_id not in demo_user_contexts:
            raise ValueError(f"Unknown user_id: {user_id}")
        self.user_id = user_id
        self.context = self._context_for_user(user_id).copy()
        return self.context.copy()

    @staticmethod
    def _context_for_user(user_id):
        if int(user_id) in demo_user_contexts:
            return demo_user_contexts[int(user_id)]
        return user_contexts[int(user_id)]

    def predict_probabilities(self, user_id, item_ids=None):
        """Return frozen, train-derived oracle probabilities for movie IDs."""
        if item_ids is None:
            item_ids = candidate_movies
        # The user's liked-genre profile makes the response depend on context;
        # the popularity term is fixed from training likes for each candidate.
        context = self._context_for_user(int(user_id))
        movie_vectors = np.vstack(
            [movie_genre_vectors[int(item_id)] for item_id in item_ids]
        )
        genre_overlap = np.sum(movie_vectors * context, axis=1)
        popularity = np.array(
            [movie_popularity_score.get(int(item_id), 0.0) for item_id in item_ids]
        )
        scores = (
            ORACLE_GENRE_WEIGHT * genre_overlap
            + ORACLE_POPULARITY_WEIGHT * popularity
            + ORACLE_INTERCEPT
        )
        scores = np.clip(scores, -700, 700)
        return 1.0 / (1.0 + np.exp(-scores))

    def step(self, action):
        if self.user_id is None:
            raise RuntimeError("Call reset(user_id) before choosing a movie.")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < len(
            candidate_movies
        ):
            raise ValueError(
                f"action must be a candidate index from 0 through {len(candidate_movies) - 1}"
            )
        item_id = candidate_movies[int(action)]
        probability = float(self.predict_probabilities(self.user_id, [item_id])[0])
        reward = int(self.rng.binomial(1, probability))
        return reward
