from flask import Flask, render_template, request
from nmf_recommend import nmf_recommender
from spiced_recommender import Recommender
from extract_infos import omdb_extract, postgres_extract
<<<<<<< HEAD
from cosine import cosine_similarity
=======
from line_graph import line

>>>>>>> 157d991d55db90fce2357e6106698aae5be2f0f9

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/hello/<name>")
def hello(name):
    name = name.upper()
    return render_template("hello.html", name_html=name)


@app.route("/movie_details/<movie_id>")
def movie_details(movie_id):
    movie_id = int(movie_id)
    results = postgres_extract(movie_id)
    return render_template("movie_details.html", results=results)


@app.route("/graph/<avg_or_total>/<movie_id>")
def movie_graph(avg_or_total, movie_id):
    movie_id = int(movie_id)
    title = postgres_extract(movie_id)[1]
    line(movie_id, title, avg_or_total)
    return render_template("graph.html")



@app.route("/recommendation")
def recommend():
    user_input = dict(request.args)
    method_ =  list(user_input.values())[-1]
    if method_ == "NMF":
        # recommendations = nmf_recommender(user_input)
        recommender = Recommender(user_input)
        recommendations = recommender.nmf()
    if  method_ == "Cosine":
        recommendations = cosine_similarity(user_input)
    for recommended_movie_id in recommendations.keys():
        postgres_infos = postgres_extract(recommended_movie_id)
        imdb_id = postgres_infos[-1]
        recommendations[recommended_movie_id]["omdb_dict"] = omdb_extract(imdb_id)
    return render_template(
        "recommendation.html", movies=recommendations, input=user_input
    )


@app.route("/recommendation_test")
def recommend_test():
    user_rating = dict(request.args)
    return render_template("recommendation_test.html", user_rating=user_rating)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
