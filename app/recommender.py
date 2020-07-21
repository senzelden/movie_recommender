import random

MOVIES = ['Toy Story 1',
          'Toy Story 2',
          'Toy Story 3',
          'Shawshank Redemption',
          'SAW',
          'The Godfather',
          'Inception',
          'Some awesome Nicolas Cage movie',
          'Python: The Movie',
          'Monty Python: Life of Brian']

def random_recommend(movies, num):
    random.shuffle(movies)
    return movies[:num]
