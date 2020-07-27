# Project: Movie Recommender

![movie_recommender](data/movie_recommender.gif)

### Description

This project implements a movie recommender using NMF and Cosine Similarity. For this the popular [Movielens](https://grouplens.org/datasets/movielens/) Dataset (smallest 100K version) is stored in a [Postgres](https://www.postgresql.org/) Database. Dash is used to visualize informations on every movie (average rating and total ratings over time, ratings per user). Fake user names are generated using the [Faker](https://github.com/joke2k/faker) library.

### Background

Companies like Netflix or Amazon prominently use recommendation systems to generate more user-oriented suggestions. [Non-Negative Matrix Factorization](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization) and [Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering) are two techniques used by recommender systems that are implemented in this project.
