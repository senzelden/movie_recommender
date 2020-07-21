from credentials import APIKEY
import requests


def movie_infos(imdb_id):
    response = requests.get(f"http://www.omdbapi.com/?i=tt{imdb_id}&apikey={APIKEY}")
    if response.status_code == 200:
        movie_dict = response.json()
    else:
        movie_dict = {}
    return movie_dict

if __name__ == '__main__':
    movie_infos(4073790)