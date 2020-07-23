import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.decomposition import NMF


def recommender(user_input):
    R_filled = pd.read_csv('../data/r_fill.csv', index_col='userId')
    m = load('../data/nmf_model.joblib')
    Q = m.components_
    new_user_vector = pd.DataFrame(R_filled.mean(axis=0).to_list(), index=R_filled.columns).transpose()
    print(new_user_vector.loc[:, '2571'])
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