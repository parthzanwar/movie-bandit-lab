"""Print a few values so a student can see the environment is wired up."""

from environment import MovieBanditEnvironment, candidate_movies


# A seeded environment makes the example repeatable if we later sample rewards.
env = MovieBanditEnvironment(random_state=7)


# Show taste vectors for three real users. Each position matches one genre in
# environment.GENRE_NAMES, and the values come only from their training likes.
for user_id in (1, 2, 3):
    context = env.reset(user_id)
    print(f"User {user_id} context: {context.round(3).tolist()}")


# Show predicted like chances for the first five fixed candidate movies for
# one user. These probabilities come from the frozen training-only oracle.
user_id = 1
env.reset(user_id)
movie_ids = candidate_movies[:5]
probabilities = env.predict_probabilities(user_id, movie_ids)
print(f"\nOracle probabilities for user {user_id}:")
for item_id, probability in zip(movie_ids, probabilities):
    print(f"  movie {item_id}: p(like) = {probability:.3f}")
