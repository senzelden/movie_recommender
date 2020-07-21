import pandas as pd
from sklearn.decomposition import NMF
from joblib import dump


def train_model():
    """returns movie recommendations dictionary with movie titles and predicted ratings based on nmf model"""
    # Read the data
    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")

    # Create Rtrue
    df = pd.merge(ratings, movies, "outer", on="movieId")
    rtrue = df[["userId", "movieId", "rating"]].set_index("userId")
    rtrue = rtrue.pivot(index=rtrue.index, columns="movieId").copy()
    rtrue_fill = rtrue.fillna(2.5).copy()

    # Run model
    m = NMF(12)
    m.fit(rtrue_fill)  # Slowest part of the code
    dump(m, '../data/nmf_model.joblib')


if __name__ == "__main__":
    train_model()