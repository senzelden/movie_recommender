import requests
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from credentials import APIKEY, PG_PASSWORD, PG_USER, PG_URL


def omdb_extract(imdb_id, info_type="Full"):
    """
    returns infos from omdb API

    PARAMS:
    -------
    imdb_id = id from imdb without prefix 'tt'
    info_type = default 'Full' (returns complete dict from omdb), 'Poster' (returns link), 'Ratings' (dict)
    """
    if len(str(imdb_id)) < 7:
        imdb_id = (7 - len(str(imdb_id))) * "0" + str(imdb_id)
    response = requests.get(f"http://www.omdbapi.com/?i=tt{imdb_id}&apikey={APIKEY}")
    movie_dict = response.json()
    if info_type == "Poster":
        return movie_dict["Poster"]
    elif info_type == "Ratings":
        return movie_dict["Ratings"]
    else:
        return movie_dict


def postgres_extract(movie_ids):
    """returns list of imdb_ids from postgres"""
    DATABASE_USER = PG_USER
    DATABASE_PASSWORD = PG_PASSWORD
    DATABASE_HOST = PG_URL
    DATABASE_PORT = "5432"
    DATABASE_DB_NAME = "movielens"

    conn = f"postgres://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_DB_NAME}"
    engine = create_engine(conn, encoding="latin1", echo=True)
    Base = declarative_base(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    class Links(Base):
        __tablename__ = "links"
        __table_args__ = {"autoload": True}

    result = session.query(Links).filter(Links.movie_id.in_(movie_ids)).all()

    imdb_ids_dict = {}
    for r in result:
        imdb_ids_dict[r.movie_id] = r.imdb_id

    return imdb_ids_dict


if __name__ == "__main__":
    print(omdb_extract(4073790))
    print(postgres_extract(109633))
