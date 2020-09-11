import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go


def line(movie_id, title, values='total'):
    """creates plotly line graph as html"""
    # prepare dataframes
    movies = pd.read_csv('../data/ml-latest-small/movies.csv')
    ratings = pd.read_csv('../data/ml-latest-small/ratings.csv')

    movies_ratings = pd.merge(movies, ratings, 'left', on='movieId')
    movies_ratings.timestamp = pd.to_datetime(movies_ratings.timestamp).copy()
    movies_ratings['year'] = movies_ratings.timestamp.dt.year

    avg_rating = movies_ratings.groupby(['movieId', 'year'])[['rating']].mean().sort_values(by='rating',
                                                                                            ascending=False)
    avg_rating.rename(columns={'rating': 'avg_rating'}, inplace=True)
    avg_rating.reset_index(inplace=True)
    avg_rating.year = avg_rating.year.astype(int)
    avg_rating.sort_values(by='year', inplace=True)

    total_ratings = movies_ratings.groupby(['movieId', 'year'])[['rating']].count().sort_values(by='rating',
                                                                                                ascending=False)
    total_ratings.rename(columns={'rating': 'total_ratings'}, inplace=True)
    total_ratings.reset_index(inplace=True)
    total_ratings.year = total_ratings.year.astype(int)
    total_ratings.sort_values(by='year', inplace=True)
    total_ratings['cumsum'] = total_ratings['total_ratings'].groupby(total_ratings['movieId']).cumsum()

    # Create a trace
    fig = go.Figure()
    if values == 'avg':
        fig.add_trace(go.Scatter(
            x=avg_rating[avg_rating.movieId == movie_id]['year'],
            y=avg_rating[avg_rating.movieId == movie_id]['avg_rating'],
            mode='lines+markers',
            name='lines',
            connectgaps=True)
        )
        fig.update_layout(title=f'{title}: Average Rating over years',
                          xaxis_title='Year',
                          yaxis_title='Average Rating')
    else:
        fig.add_trace(trace = go.Scatter(
            x=total_ratings[total_ratings.movieId == movie_id]['year'],
            y=total_ratings[total_ratings.movieId == movie_id]['cumsum'],
            mode='lines+markers',
            name='lines',
            connectgaps=True)
        )
        fig.update_layout(title=f'{title}: Cumulative total ratings over years',
                          xaxis_title='Year',
                          yaxis_title='Total Ratings')
    fig.write_html("templates/graph.html")

if __name__ == "__main__":
    print(line(1))
