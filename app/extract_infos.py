import requests
from credentials import APIKEY, PG_PASSWORD, PG_USER, PG_URL
from sqlalchemy import create_engine, text
import pandas as pd


def omdb_extract(imdb_id, info_type="Full"):
    """
    returns infos from omdb API

    PARAMS:
    -------
    imdb_id = id from imdb without prefix 'tt'
    info_type = default 'Full' (returns complete dict from omdb), 'Poster' (returns link), 'Ratings' (dict)
    """
    response = requests.get(f"http://www.omdbapi.com/?i=tt{imdb_id}&apikey={APIKEY}")
    movie_dict = response.json()
    if info_type == "Poster":
        return movie_dict["Poster"]
    elif info_type == "Ratings":
        return movie_dict["Ratings"]
    else:
        return movie_dict


def postgres_extract(movie_id):
    """returns tuple (movie_id, title, genre, avg_rating, total_ratings, imdb_id) from postgres"""
    links = pd.read_csv("../data/ml-latest-small/links.csv", converters={'imdbId': lambda x: str(x)})
    return links[links.movieId == int(movie_id)].imdbId.values[0]
    # conns = f"postgres://{PG_USER}:{PG_PASSWORD}@{PG_URL}/movie_recommender"
    # db = create_engine(conns, encoding="UTF-8", echo=False)
    # query = """
    # SELECT movies.movie_id, movies.title, movies.genre, round(avg(ratings.rating)::numeric,2) AS avg_rating, count(ratings.rating) AS ratings_total, links.imdbid
	# FROM movies
    # 	LEFT JOIN ratings ON movies.movie_id = ratings.movie_id
	#  		LEFT JOIN links ON movies.movie_id = links.movie_id
	# WHERE movies.movie_id = :movie_id
    # GROUP BY movies.movie_id, movies.title, links.imdbid
    # """
    # row = db.execute(text(query), {"movie_id": str(movie_id)}).fetchone()
    # return row


if __name__ == "__main__":
    print(omdb_extract(4073790))
    print(postgres_extract(109633))
