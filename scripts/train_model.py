import pandas as pd
from joblib import dump
from sklearn.decomposition import NMF
from sklearn.impute import KNNImputer



def train_model(n_features=20, fill_method='KNNimputer', fill_value=2.5):
    """
    returns movie recommendations dictionary with movie titles and predicted ratings based on nmf model

    PARAMS
    -------
    n_features (optiona): number of features to be used for NMF (default is 20)
    fill_method (optional): either 'single_value' or 'KNNimputer' (default)
    fill_value (optional): if 'single_value' is used as fill_method, fill_value can be set manually (e.g. 0, 2.5, 3)
    """
    # Read the data
    movies = pd.read_csv("../data/ml-latest-small/movies.csv")
    ratings = pd.read_csv("../data/ml-latest-small/ratings.csv")

    # Create Rtrue
    df = pd.merge(ratings, movies, how="left", on="movieId")
    rtrue = df[["userId", "movieId", "rating"]].set_index("userId")
    rtrue = rtrue.pivot(index=rtrue.index, columns="movieId").copy()
    if fill_method == 'KNNimputer':
        imputer = KNNImputer(n_neighbors=5)
        rtrue_fill = pd.DataFrame(imputer.fit_transform(rtrue), columns=rtrue.columns, index=rtrue.index)
    else:
        rtrue_fill = rtrue.fillna(fill_value).copy()

    # Run model
    m = NMF(n_features, max_iter=1000)
    m.fit(rtrue_fill)  # Slowest part of the code
    dump(m, "../data/nmf_model.joblib")


if __name__ == "__main__":
    train_model()
