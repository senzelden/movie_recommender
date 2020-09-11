from flask import Flask, render_template, request

from recommender import Recommender
from recommender_with_spark import sparkRecommender
from extract_infos import omdb_extract, postgres_extract


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommendation")
def recommend():
    user_input = dict(request.args)
    method_ = list(user_input.values())[-1]
    recommender = Recommender(user_input)
    spark_recommender = sparkRecommender(user_input)
    if method_ == "NMF":
        recommendations = recommender.nmf()
    elif method_ == "Cosine":
        recommendations = recommender.cosine()
    else:
        recommendations = spark_recommender.als()
    imdb_ids_dict = postgres_extract(recommendations.keys())
    for movie_id, imdb_id in imdb_ids_dict.items():
        recommendations[movie_id]["omdb_dict"] = omdb_extract(imdb_id)
    return render_template(
        "recommendation.html", movies=recommendations, input=user_input
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
