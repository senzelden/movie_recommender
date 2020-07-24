import numpy as np
import pandas as pd
from joblib import load


class Recommender:

    def __init__(self, user_input):
        self.user_input = user_input

    def nmf(self):
        movie_ids = load('../data/movie_ids.joblib')
        r_fill_mean = load('../data/R_fill_mean.joblib')
        m = load('../data/nmf_model.joblib')
        new_user_vector = pd.DataFrame(r_fill_mean, index=movie_ids).transpose()
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
        ]  # list of movie indices based on movie_id
        movies_user_has_seen = []
        for i, user_movie_id in enumerate(user_input_ids):
            if f"seen{i + 1}" in self.user_input.keys():
                movies_user_has_seen.append(user_movie_id)
                current_rating = int(self.user_input[f"rating{i + 1}"])
                if current_rating > 40:
                    multiplier = 10
                elif current_rating > 30:
                    multiplier = 1
                elif current_rating > 20:
                    multiplier = 0.1
                else:
                    multiplier = 0.001
                new_user_vector.loc[:, str(user_movie_id)] = current_rating * multiplier
        hidden_profile = m.transform(new_user_vector)
        rating_predictions = pd.DataFrame(np.dot(hidden_profile, m.components_), columns=movie_ids)
        # Create a boolean mask to filter for already seen movies
        seen_movies_indices = []
        for movie_seen_id in movies_user_has_seen:
            seen_movies_indices.append(np.where(movie_ids == str(movie_seen_id))[0][0])
        bool_mask = [True] * len(movie_ids)
        for index in seen_movies_indices:
            bool_mask[index] = False
        # bool_mask = [False if int(column) in movies_user_has_seen else True for column in movie_ids]
        movies_not_seen = rating_predictions.columns[bool_mask]
        movies_not_seen_df = rating_predictions[movies_not_seen].T
        sorted_movies = movies_not_seen_df.sort_values(by=0, ascending=False)[:3]
        recommendations = {}
        for i in range(3):
            recommendations[int(sorted_movies.index[i])] = {"nmf_score": round(sorted_movies.values[i][0], 2)}

        return recommendations


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
