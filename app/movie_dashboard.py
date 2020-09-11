# -*- coding: utf-8 -*-
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd
import plotly.graph_objects as go

from credentials import PG_USER, PG_PASSWORD, PG_URL


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

DATABASE_USER = PG_USER
DATABASE_PASSWORD = PG_PASSWORD
DATABASE_HOST = PG_URL
DATABASE_PORT = "5432"
DATABASE_DB_NAME = "movielens"

conn = f"postgres://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_DB_NAME}"
movies = pd.read_sql_table('movies', conn)
ratings = pd.read_sql_table('ratings_2019_w_timestamp', conn)

movies_ratings = pd.merge(movies, ratings, "left", on="movie_id")
movies_ratings.rating_timestamp = pd.to_datetime(movies_ratings.rating_timestamp).copy()
movies_ratings["month"] = movies_ratings.rating_timestamp.dt.month
movies_ratings.user_id = movies_ratings.user_id.astype(pd.Int32Dtype())

avg_rating = (
    movies_ratings.groupby(["movie_id", "month"])[["rating"]]
    .mean()
    .sort_values(by="rating", ascending=False)
)
avg_rating.rename(columns={"rating": "avg_rating"}, inplace=True)
avg_rating.reset_index(inplace=True)
avg_rating.sort_values(by="month", inplace=True)

total_ratings = (
    movies_ratings.groupby(["movie_id", "month"])[["rating"]]
    .count()
    .sort_values(by="rating", ascending=False)
)
total_ratings.rename(columns={"rating": "total_ratings"}, inplace=True)
total_ratings.reset_index(inplace=True)
total_ratings.sort_values(by="month", inplace=True)
total_ratings["cumsum"] = (
    total_ratings["total_ratings"].groupby(total_ratings["movie_id"]).cumsum()
)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=avg_rating[avg_rating.movie_id == 318]["month"],
        y=avg_rating[avg_rating.movie_id == 318]["avg_rating"],
        mode="lines+markers",
        name="lines",
        connectgaps=True,
    )
)
fig.update_layout(
    title="Average Rating over months", xaxis_title="Month", yaxis_title="Average Rating"
)

fig2 = go.Figure()
fig2.add_trace(
    trace=go.Scatter(
        x=total_ratings[total_ratings.movie_id == 318]["month"],
        y=total_ratings[total_ratings.movie_id == 318]["cumsum"],
        mode="lines+markers",
        name="lines",
        connectgaps=True,
    )
)
fig2.update_layout(
    title="Cumulative total ratings over months",
    xaxis_title="Month",
    yaxis_title="Total Ratings",
)

app.layout = html.Div(
    children=[
        html.H1(
            children="THE SHAWSHANK REDEMPTION (1994)", style={"textAlign": "center"}
        ),
        html.Div(
            children=[
                dcc.Graph(figure=fig, id="avg-graph"),
                dcc.Graph(figure=fig2, id="total-graph"),
                dash_table.DataTable(
                    id="table",
                    columns=[
                        {"name": i, "id": i}
                        for i in movies_ratings[movies_ratings.movie_id == 318][
                            ["user_id", "rating", "rating_timestamp"]
                        ].columns
                    ],
                    data=movies_ratings[movies_ratings.movie_id == 318][
                        ["user_id", "rating", "rating_timestamp"]
                    ].to_dict("records"),
                    sort_mode="single",
                    sort_action="native",
                    fixed_rows={"headers": True},
                    style_as_list_view=True,
                    style_cell={
                        "minWidth": 95,
                        "maxWidth": 95,
                        "width": 95,
                        "padding": "5px",
                    },
                    style_header={
                        "backgroundColor": "lightgreen",
                        "fontWeight": "bold",
                        "border": "1px solid black",
                    },
                    style_cell_conditional=[{"if": {}, "textAlign": "left"}],
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "rgb(248, 248, 248)",
                        }
                    ],
                ),
            ]
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
