from joblib import dump
import pandas as pd
from sklearn.decomposition import NMF

from credentials import PG_USER, PG_PASSWORD, PG_URL


# Establish connection to database and read tables
DATABASE_USER = PG_USER
DATABASE_PASSWORD = PG_PASSWORD
DATABASE_HOST = PG_URL
DATABASE_PORT = "5432"
DATABASE_DB_NAME = "movielens"

conn = f"postgres://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_DB_NAME}"
ratings_2019 = pd.read_sql_table("ratings_2019", conn)
movies_ratings_2019 = pd.read_sql_table("movies_ratings_2019", conn)
popular_movies = movies_ratings_2019[
    movies_ratings_2019.total_ratings >= 5
].movie_id.values
filtered_ratings_2019 = ratings_2019[ratings_2019.movie_id.isin(popular_movies)]
print("Ratings table has been loaded and filtered.")

# Prepare matrix
rtrue = filtered_ratings_2019
rtrue = rtrue.pivot(index="user_id", columns="movie_id")
print("Shape of R: ", rtrue.shape)
n = 1000  # Chunk row size
list_df = [rtrue[i : i + n] for i in range(0, rtrue.shape[0], n)]
for i in range(len(list_df)):
    list_df[i] = list_df[i].fillna(2.5)
rtrue_fill = pd.concat(list_df)
print("Shape of R: ", rtrue.shape)

# Train model
m = NMF(20, max_iter=100)
m.fit(rtrue_fill)  # Slowest part of the code
print("Model has been trained.")

# Save model
# dump(m, "../data/nmf_model.joblib")
# print('Model has been saved.')
