from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent / "data" / "ml-100k"

ratings = pd.read_csv(
    DATA_DIR / "u.data",
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"],
)

genre_names = [
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
movie_columns = [
    "item_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    *genre_names,
]
movies = pd.read_csv(
    DATA_DIR / "u.item",
    sep="|",
    names=movie_columns,
    encoding="latin-1",
)

print(f"Number of users: {ratings['user_id'].nunique()}")
print(f"Number of movies: {len(movies)}")
print(f"Number of ratings: {len(ratings)}")
print("\nRating distribution:")
print(ratings["rating"].value_counts().sort_index().to_string())

most_rated = (
    ratings.groupby("item_id").size().rename("rating_count").reset_index()
    .merge(movies[["item_id", "title"]], on="item_id", how="left")
    .sort_values(["rating_count", "item_id"], ascending=[False, True])
    .head(10)
)
print("\n10 most-rated movies:")
print(most_rated[["title", "rating_count"]].to_string(index=False))
