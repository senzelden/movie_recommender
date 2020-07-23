import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity(user_input):
    """
    It returns a cosine transformed data frame with pairwise sililarity in ratings between users

    Parameters
    ----------
    r_true corresponds to the either imputed or filled NaNs R table.
    """

    r_true = pd.read_csv('../data/R_table.csv')
    users = r_true.index
    rtrue_fill.columns = rtrue_fill.columns.droplevel(0)

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
    for film in landing_page_movies:
        indices.append(rtrue_fill.columns.get_loc(film))
    for i, indices_value in enumerate(indices):
        if f"seen{i + 1}" in user_input.keys():
            new_user[indices_value] = user_input[f"rating{i + 1}"]

    new_user_filled = new_user.fillna(2.5)
    new_user_final = np.array([new_user_filled])



    try:
        r_cosine = pd.DataFrame(cosine_similarity(r_true), columns=users, index=users)
    except:
        print('The cosine_similarity function from sklearn cannot handel missing values')
    return r_cosine
