import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity as cos_similarity


def cosine_similarity(user_input):
    movies = pd.read_csv("../data/ml-latest-small/movies.csv")
    ratings = pd.read_csv("../data/ml-latest-small/ratings.csv")
    R_filled = pd.read_pickle('../data/R_filled_KNN_binary')
    movies_ratings = pd.merge(movies, ratings, on="movieId")
    user_input_ids = [
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
    ]
    new_user_vector = pd.DataFrame([np.nan] * len(R_filled.columns), index=R_filled.columns).transpose()
    movies_user_has_seen = []
    for i, user_movie_id in enumerate(user_input_ids):
        if f"seen{i + 1}" in user_input.keys():
            movies_user_has_seen.append(user_movie_id)
            current_rating = int(user_input[f"rating{i + 1}"])
            new_user_vector.loc[:, user_movie_id] = current_rating / 10
    R_new_user = R_filled.append(new_user_vector)
    movie_filter = ~R_new_user.isna().any().values
    updated_users = R_new_user.index
    similarities_new_user = pd.DataFrame(cos_similarity(R_new_user.transpose()[movie_filter].transpose()), \
                                         index=updated_users, columns=updated_users)
    similarities_0 = similarities_new_user[0][~(similarities_new_user.index == 0)]
    rating_predictions = pd.DataFrame(np.dot(similarities_0, R_filled) / similarities_0.sum(), index=R_filled.columns)
    best_predictions = rating_predictions[~movie_filter].sort_values(by=0, ascending=False)[:300]
    best_preds = best_predictions.index.to_list()
    movies_titles_genres = movies_ratings.groupby(['movieId', 'title', 'genres'])[['rating']].count()
    movies_titles_genres.reset_index(inplace=True)
    movies_titles_genres.set_index('movieId', inplace=True)
    possible_recommendations = movies_titles_genres.filter(best_preds, axis=0)
    highest_rated_by_user = new_user_vector.transpose().idxmax().to_list()
    all_genres = movies_titles_genres.filter(highest_rated_by_user, axis=0).genres.values[0]
    recommendations_df = possible_recommendations[
        possible_recommendations.genres.str.contains(all_genres, regex=True)].sort_values('rating',
                                                                                          ascending=False).head(3)
    recommendations = {}
    for i in range(3):
        top_movie_id = recommendations_df.index[i]
        recommendations[int(top_movie_id)] = {"Cosine": round(best_predictions.loc[top_movie_id].values[0], 2)}
    return recommendations

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
    print(cosine_similarity(example_input))
