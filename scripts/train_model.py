import pandas as pd
from joblib import dump
from sklearn.decomposition import NMF
from sklearn.impute import KNNImputer

def create_r(movies, ratings):
    """
    Creates a data frame with the information necessary for running an NMF model for movie recommendations

    Parameters:
    ----------
    movies csv file containing information about movieId, title, genres
    ratings csv file containing information about movieId, userId, rating, timestamp
    """
    # Create r_true
    df = pd.merge(ratings, movies, how="left", on="movieId")
    r_true = df[["userId", "movieId", "rating"]].set_index("userId")
    r_true = r_true.pivot(index=r_true.index, columns="movieId").copy()
    return r_true

def imputeKNN(r_true, neighbors):
    """
    Uses k-nearest neighbors in the space of user to impute missing rating values

    Parameters:
    ----------
    r_true is the data frame with missing values in which imputing the values is required
    """
    R = r_true
    movies = R.columns
    users = R.index
    R = R.replace(r'NaN', np.nan, regex=True)
    imputer = KNNImputer(n_neighbors=neighbors)
    R_imputed = pd.DataFrame(imputer.fit_transform(R), columns=movies, index=users)
    return R_imputed


def train_model(n_features=20, fill_method = "KNNimputer", fill_value=2.5):
    """
    Creates a Dictionary with movie titles and predicted ratings based on Search Results Non-negative
    matrix factorization model (NMF). To do so, it either uses a k-nearest neighbors, or a fill in
    with a user define value

    Parameters:
    ----------
    For model: n_features (optional): number of features to be used for NMF (default is 20)
    For model: fill_method (optional): either an empty string or 'KNNimputer' (default).
    fill_value (optional): if 'single_value' is used as fill_method.Fill_value can be set manually (e.g. 0, 2.5, 3)
    """

    if fill_method == 'KNNImputer':
        r_true_fill = imputeKNN(r_true, n_features)
    else:
        r_true_fill = r_true.fillna(fill_value).copy()
    r_true_fill.to_csv('../data/R_table.csv', sep =',')
    m = NMF(n_features, max_iter=100000)
    m.fit(r_true_fill)
    dump(m, "../data/nmf_model.joblib")


# Load the data
movies = pd.read_csv("../data/ml-latest-small/movies.csv")
ratings = pd.read_csv("../data/ml-latest-small/ratings.csv")

if __name__ == "__main__":
    r_true = create_r(movies, ratings)
    train_model()
