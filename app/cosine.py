import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity as cos_similarity
from sklearn.impute import KNNImputer
from train_model import imputeKNN

def cosine_similarity(user_input):
    """
    It returns a cosine transformed data frame with pairwise sililarity in ratings between users

    Parameters
    ----------
    r_true corresponds to the either imputed or filled NaNs R table.
    """

    rtrue_fill = pd.read_csv("../data/R_table_2.csv", index_col = 0)
    rtrue_fill = rtrue_fill.tail(50) # ONly consider 50 users for imputtin KNNN
    users = rtrue_fill.index
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
    ]
    indices = []
    new_user = {}
    for film in landing_page_movies:
        index_val = rtrue_fill.columns.get_loc(str(film))
        indices.append(index_val)
    for i, indices_value in enumerate(indices):
        if user_input[f"seen{i + 1}"] == "True":
            new_user[str(indices_value)] = user_input[f"rating{i + 1}"]
    print(new_user)
    R_new_user = pd.DataFrame(new_user, index=['new_user'])
    R_new_user.columns = [str(i) for i in landing_page_movies]
    R_new_user = rtrue_fill.append(R_new_user)
    updated_users = R_new_user.index
    new_user_filled = imputeKNN(R_new_user, 20)

    movie_filter = ~R_new_user.isna().any().values
    similarities_new_user = cos_similarity(new_user_filled.transpose()[movie_filter].transpose())

    similarities_new_user = pd.DataFrame(similarities_new_user, index=updated_users, columns=updated_users)

    similarities_new_user = similarities_new_user['new_user'][~(similarities_new_user.index=='new_user')]
    rating_predictions = pd.DataFrame(\
                            np.dot(similarities_new_user, rtrue_fill)
                            /similarities_new_user.sum(), \
                            index=rtrue_fill.columns)

    recommendations = rating_predictions[~movie_filter].sort_values(by=0, ascending=False)
    recommendations = recommendations[0:3]
    print('This is the sahepe of recommendations :', recommendations.shape)
    indexes = list(recommendations.index)
    indexes = [int(i) for i in indexes]
    print(indexes)
    for i in range(3):
        print(indexes[i])
        recommendations[int(recommendations.index[i])] = {"Cosine": values[i]}
    print(recommendations)

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
