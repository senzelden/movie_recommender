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
    rtrue_fill = rtrue_fill.tail(50)
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
        indices.append(rtrue_fill.columns.get_loc(str(film)))
    for i, indices_value in enumerate(indices):
         if f"seen{i + 1}" in user_input.keys():
            new_user[indices_value] = user_input[f"rating{i + 1}"]
    R_new_user = rtrue_fill.append(pd.DataFrame(new_user, index=['new_user']))
    R_new_user = R_new_user
    #print("This is R new user", R_new_user) # I should Impute here
    new_user_filled = imputeKNN(R_new_user, 20)
    movie_filter = R_new_user.isna().any().values
    updated_users = R_new_user.index
    print("THIS IS MOVIE FILTER:", movie_filter.shape)

#    THIS IS MOVIE FILTER: (9734,)
#    THIS IS RATING PREDICTIONS : (9724, 1)
#

    similarities_new_user = pd.DataFrame(cos_similarity(new_user_filled.transpose()[movie_filter].transpose()), index=updated_users, columns=updated_users)
    similarities_new_user = similarities_new_user['new_user'][~(similarities_new_user.index=='new_user')]
    print('THIS IS SIMILARITIES:', similarities_new_user.shape)
    rating_predictions = pd.DataFrame(\
                           np.dot(similarities_new_user, rtrue_fill)
                           /similarities_new_user.sum(), \
                           index=rtrue_fill.columns)

    print('THIS IS RATING PREDICTIONS :', rating_predictions.shape)
    recommendations = rating_predictions[~movie_filter].sort_values(by=0, ascending=False)
    #recommendations = recommendations[0:3]
    #print(recommendations)
    #return suggestions

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
