import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from joblib import load


def nmf_recommender(user_input):
    """returns movie recommendations dictionary with movie titles and predicted ratings based on nmf model"""
    # Read the data
    movies = pd.read_csv("../data/ml-latest-small/movies.csv")
    ratings = pd.read_csv("../data/ml-latest-small/ratings.csv")

    # Create Rtrue
    df = pd.merge(ratings, movies, how="left", on="movieId")
    rtrue = df[["userId", "movieId", "rating"]].set_index("userId")
    rtrue = rtrue.pivot(index=rtrue.index, columns="movieId").copy()
    imputer = KNNImputer(n_neighbors=5)
    rtrue_fill = pd.DataFrame(imputer.fit_transform(rtrue), columns=rtrue.columns, index=rtrue.index)

    # Load trained model
    m = load("../data/nmf_model.joblib")
    P = m.components_
    # Q = m.transform(rtrue_fill)

    # Initiate new_user
    new_user = [3.5] * 9724

    # Example ratings
    landing_page_movies = [
        2571,
        356,
        318,
        2160,
        899,
        8464,
        2959,
        68954,
        4993,
        296,
    ]  # list of movie indices based on movie_id
    rtrue_fill.columns = rtrue_fill.columns.droplevel(0)
    indices = []
    for film in landing_page_movies:
        indices.append(rtrue_fill.columns.get_loc(film))
    for i, indices_value in enumerate(indices):
        if f"seen{i + 1}" in user_input.keys():
            new_user[indices_value] = int(user_input[f"rating{i + 1}"]) / 10
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
        if (index not in landing_page_movies) and (len(clean_recommendations) < 3):
            # movies[movies.movieId == index].title.values[0]
            clean_recommendations[index] = {"nmf_score": round(score, 2)}
    return clean_recommendations


if __name__ == "__main__":
    example_input = {
        "seen1": "True",
        "rating1": "15",
        "seen2": "True",
        "rating2": "22",
        "seen3": "True",
        "rating3": "32",
        "seen4": "True",
        "rating4": "28",
        "seen5": "True",
        "rating5": "25",
        "seen6": "True",
        "rating6": "25",
        "seen7": "True",
        "rating7": "49",
        "seen8": "True",
        "rating8": "50",
        "seen9": "True",
        "rating9": "34",
        "seen10": "True",
        "rating10": "50",
    }
    print(nmf_recommender(example_input))
