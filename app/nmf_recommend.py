import numpy as np
import pandas as pd
from sklearn.decomposition import NMF


def nmf_recommender(user_ratings=(5, 5, 5, 5, 5, 5, 5, 5, 5, 5)):
    """returns movie recommendations dictionary with movie titles and predicted ratings based on nmf model"""
    # Read the data
    movies = pd.read_csv("../data/ml-latest-small/movies.csv")
    ratings = pd.read_csv("../data/ml-latest-small/ratings.csv")

    # Create Rtrue
    df = pd.merge(ratings, movies, "outer", on="movieId")
    rtrue = df[["userId", "movieId", "rating"]].set_index("userId")
    rtrue = rtrue.pivot(index=rtrue.index, columns="movieId").copy()
    rtrue_fill = rtrue.fillna(2.5).copy()

    # Run model
    m = NMF(12)
    m.fit(rtrue_fill)  # Slowest part of the code

    P = m.components_
    # Q = m.transform(rtrue_fill)

    # Initiate new_user
    new_user = [2.5] * 9742

    # Example ratings
    standard_movies = [
        442,
        508,
        153,
        567,
        311,
        53,
        251,
        515,
        25,
        30,
    ]  # list of movie indices based on movie_id
    rtrue_fill.columns = rtrue_fill.columns.droplevel(0)
    indices = []
    for film in standard_movies:
        indices.append(rtrue_fill.columns.get_loc(film))
    for i, indices_value in enumerate(indices):
        new_user[indices_value] = user_ratings[i]
    new_user_final = np.array([new_user])

    # Get recommendations for user
    user_profile = m.transform(new_user_final)
    result = np.dot(user_profile[0], P)
    new_result = pd.DataFrame(result)
    new_result = new_result.transpose().copy()
    new_result.columns = rtrue_fill.columns
    recommendations = new_result.iloc[0].sort_values(ascending=False)[:20].to_dict()
    clean_recommendations = {}
    for index, score in recommendations.items():
        if index not in standard_movies:
            clean_recommendations[
                movies[movies.movieId == index].title.values[0]
            ] = round(score, 2)
    return clean_recommendations


if __name__ == "__main__":
    print(nmf_recommender())
