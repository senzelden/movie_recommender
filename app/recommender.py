from joblib import load
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as cos_similarity


class Recommender:
    def __init__(self, user_input):
        self.user_input = user_input
        self.user_input_ids = [
            858,
            63992,
            58559,
            1924,
            2324,
            171011,
            177765,
            296,
            5618,
            1136,
        ]  # list of movie id's from landing page

    def nmf(self):
        """
        Non-Negative Matrix Factorization -
        unsupervised learning algorithm, implemented here to return movie recommendations based on user ratings
        """

        def update_vector(self, mean_rating_vector):
            """updates user vector with new user ratings, returns new vector and list of seen movies"""
            vector = mean_rating_vector.transpose()
            movies_user_has_seen = []
            for i, user_movie_id in enumerate(self.user_input_ids):
                if f"seen{i}" in self.user_input.keys():
                    movies_user_has_seen.append(user_movie_id)
                    current_rating = int(self.user_input[f"rating{i}"])
                    if current_rating > 40:
                        multiplier = 10
                    elif current_rating > 30:
                        multiplier = 1
                    elif current_rating > 20:
                        multiplier = 0.1
                    else:
                        multiplier = 0.001
                    vector.loc[:, user_movie_id] = current_rating * multiplier
            return vector, movies_user_has_seen

        def filter_for_movies_not_seen(movies_user_has_seen, movie_ids):
            """
            get indices of predictions df for movies seen by user,
            return a boolean mask to filter for already seen movies (not seen: True, seen: False)
            """
            seen_movies_indices = []
            for movie_seen_id in movies_user_has_seen:
                seen_movies_indices.append(np.where(movie_ids == movie_seen_id)[0][0])
            bool_mask = [True] * len(movie_ids)
            for index in seen_movies_indices:
                bool_mask[index] = False
            # bool_mask = [False if int(column) in movies_user_has_seen else True for column in movie_ids]
            return bool_mask

        # load data and model
        mean_rating_vector = load("../data/mean_rating_vector.joblib")
        m = load("../data/nmf_model.joblib")
        movie_ids = mean_rating_vector.index

        # get predictions for new user
        new_user_vector, movies_user_has_seen = update_vector(self, mean_rating_vector)
        hidden_profile = m.transform(new_user_vector)
        rating_predictions = pd.DataFrame(
            np.dot(hidden_profile, m.components_), columns=movie_ids
        )

        # filter out seen movies
        bool_mask = filter_for_movies_not_seen(movies_user_has_seen, movie_ids)
        movies_not_seen = rating_predictions.columns[bool_mask]
        movies_not_seen_df = rating_predictions[movies_not_seen].T

        # prepare recommendations
        sorted_movies = movies_not_seen_df.sort_values(by=0, ascending=False)[:3]
        recommendations = {}
        for i in range(3):
            recommendations[int(sorted_movies.index[i])] = {
                "nmf_score": round(sorted_movies.values[i][0], 2)
            }

        return recommendations

    def cosine(self, genre_filter="top2_genres"):
        """
        returns a cosine transformed dataframe with pairwise similarity in ratings between users
        """

        def update_matrix(R_filled, new_user_vector):
            """updates matrix R with new user ratings, returns new matrix and list of seen movies"""
            movies_user_has_seen = []
            for i, user_movie_id in enumerate(self.user_input_ids):
                if f"seen{i}" in self.user_input.keys():
                    movies_user_has_seen.append(user_movie_id)
                    current_rating = int(self.user_input[f"rating{i}"])
                    new_user_vector.loc[:, user_movie_id] = current_rating / 10
            R_new_user = R_filled.append(new_user_vector)
            return R_new_user, movies_user_has_seen

        def filter_by_genre(vector,recs,movie_data,genre_filter):
            """filters recommendations by genre and returns filtered dataframe"""
            highest_rated_by_user = vector.transpose().idxmax().to_list()
            if genre_filter == "any_of_all_genres":
                all_genres = movie_data.filter(
                    highest_rated_by_user, axis=0
                ).genres.values[0]
                recommendations_df = (
                    recs[
                        recs.genres.str.contains(
                            all_genres, regex=True
                        )
                    ]
                    .sort_values("rating", ascending=False)
                    .head(3)
                )
            else:
                top3_rated_by_user = vector.transpose().nlargest(3, 0).index
                top3_rated_by_user_genres = (
                    movie_data.filter(top3_rated_by_user, axis=0)
                    .genres.str.split("|")
                    .values
                )
                most_important_genres = {}
                for genre_list in top3_rated_by_user_genres:
                    for genre in genre_list:
                        if genre not in most_important_genres.keys():
                            most_important_genres[genre] = 1
                        else:
                            most_important_genres[genre] += 1
                sorted_genres = {
                    k: v
                    for k, v in sorted(
                        most_important_genres.items(),
                        key=(lambda item: item[1]),
                        reverse=True,
                    )
                }
                top2_genres = sorted(list(sorted_genres.keys())[:2])
                top2_genres = "|".join(top2_genres)
                recommendations_df = (
                    recs[
                        recs.genres.str.contains(
                            top2_genres, regex=False
                        )
                    ]
                    .sort_values("rating", ascending=False)
                    .head(50)
                )

            return recommendations_df

        # Load the data
        R_filled = load("../data/rtrue_fillna_25.joblib")
        movies_titles_genres = load("../data/movies_titles_genres.joblib")

        # Make predictions
        new_user_vector = pd.DataFrame(
            [np.nan] * len(R_filled.columns), index=R_filled.columns
        ).transpose()
        R_new_user, movies_user_has_seen = update_matrix(R_filled, new_user_vector)
        movie_filter = ~R_new_user.isna().any().values
        updated_users = R_new_user.index
        similarities_new_user = pd.DataFrame(
            cos_similarity(R_new_user.transpose()[movie_filter].transpose()),
            index=updated_users,
            columns=updated_users,
        )
        similarities_0 = similarities_new_user[0][~(similarities_new_user.index == 0)]
        rating_predictions = pd.DataFrame(
            np.dot(similarities_0, R_filled) / similarities_0.sum(),
            index=R_filled.columns,
        )
        best_predictions = rating_predictions[~movie_filter].sort_values(
            by=0, ascending=False
        )[:1000]
        best_preds = best_predictions.index.to_list()
        possible_recommendations = movies_titles_genres.filter(best_preds, axis=0)

        # Filter and prepare recommendations
        recommendations_df = filter_by_genre(
            new_user_vector,
            possible_recommendations,
            movies_titles_genres,
            genre_filter,
        )
        if len(recommendations_df) < 3:
            recommendations_df = possible_recommendations.sort_values(
                "rating", ascending=False
            ).head(3)
        recommendations = {}
        for i in range(3):
            top_movie_id = recommendations_df.index[i]
            recommendations[int(top_movie_id)] = {
                "cosine_score": round(best_predictions.loc[top_movie_id].values[0], 2)
            }
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
    recommender = Recommender(example_input)
    print(recommender.nmf())
    print(recommender.cosine())
