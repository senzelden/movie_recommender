"""Spark Recommender"""
try:
    spark.stop()
except:
    pass

import findspark

findspark.init()

import configparser
import re
from collections import Counter

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.functions import col, desc
from pyspark.ml.recommendation import ALS


class sparkRecommender:
    def __init__(self, user_input):
        self.user_input = user_input
        self.user_input_ids = [
            858,
            63992,
            58559,
            1924,
            2324,
            171011,
            177765,
            296,
            5618,
            1136,
        ]  # list of movie id's from landing page
        self.user_ratings = [2.5] * 10

    def als(self):
        """takes in user ratings from 2019 and returns predictions based on Spark's ALS model"""

        def get_db_properties():
            """loads postgresql information from config file"""
            db_properties = {}
            config = configparser.ConfigParser()
            config.read("db_properties.ini")
            db_prop = config["postgresql"]
            db_properties["user"] = db_prop["user"]
            db_properties["password"] = db_prop["password"]
            db_properties["url"] = db_prop["url"]
            db_properties["driver"] = db_prop["driver"]
            return db_properties

        def get_user_ratings(self):
            """returns list of ratings from landing page and list of seen movies by user"""
            movies_user_has_seen = []
            for i, user_movie_id in enumerate(self.user_input_ids):
                if f"seen{i}" in self.user_input.keys():
                    movies_user_has_seen.append(user_movie_id)
                    current_rating = int(self.user_input[f"rating{i}"])
                    self.user_ratings[i] = current_rating / 10
            return movies_user_has_seen

        def get_new_rtrue(self, db_ratings_2019):
            """Appends ratings from landing page to ratings table"""
            new_user_ratings = [
                (0, self.user_input_ids[i], self.user_ratings[i])
                for i in range(len(self.user_input_ids))
            ]
            new_user_df = spark.createDataFrame(
                new_user_ratings, ["userId", "movieId", "rating"]
            )
            new_rtrue = db_ratings_2019.union(new_user_df)
            return new_rtrue

        def get_recommendations_for_new_user(model, num_recommendations=500):
            """determine recommendations for selected user"""
            new_user = spark.createDataFrame([(0,)], ["userId"])
            user_subset_recs = model.recommendForUserSubset(new_user, num_recommendations)
            result = user_subset_recs.collect()
            row = result[0]
            recommended_movies = []
            for i in range(num_recommendations):
                recommended_movies.append(row.asDict()["recommendations"][i]["movieId"])
            return recommended_movies

        def get_relevant_genre(user_movies, movies):
            """find most relevant genre for new user"""
            high_rated = []
            for (key, value) in user_movies.items():
                if value > 3.5:
                    high_rated.append(key)
            user_genres = [
                row.genres
                for row in movies.filter(movies.movieId.isin(high_rated)).collect()
            ]
            words = re.findall(r"[a-zA-Z'-]+", " ".join(user_genres))
            words = sorted(words)
            important_genre = Counter(words).most_common(1)
            try:
                top_genre = important_genre[0][0]
            except:
                top_genre = "(no genres listed)"
            return top_genre

        def filter_recommendations(recommended_movies, movies_ratings_2019):
            """filter recommendations by genre and average rating, return dict with top 10 recommendations"""
            filtered_recommendations = (
                movies_ratings_2019.filter(
                    movies_ratings_2019.movieId.isin(recommended_movies)
                )
                .filter(movies_ratings_2019.genres.contains(top_genre))
                .filter(movies_ratings_2019.avg_rating > 3.5)
                .sort(desc("total_ratings"))
                .limit(10)
            )
            filtered_recommended_movies = {
                row.movieId: row.title for row in filtered_recommendations.collect()
            }
            return filtered_recommended_movies

        def output_shape(filtered_recs, movies_user_has_seen, num_recs=3):
            """reduce number of recommendations, avoid movies user has seen and return as dictionary"""
            counter = 0
            recommendations = {}
            for key, value in filtered_recs.items():
                if counter >= num_recs:
                    break
                else:
                    if key not in movies_user_has_seen:
                        print(value)
                        recommendations[int(key)] = {"title": value}
                        counter += 1
                    else:
                        pass
            return recommendations

        # Set up Spark
        conf = SparkConf()
        conf.set(
            "spark.jars",
            "../data/jars/postgresql-42.2.16.jar",
        )
        spark = (
            SparkSession.builder.appName("Spark_Recommender")
            .config(conf=conf)
            .getOrCreate()
        )

        # Load the data from PostgreSQL RDS
        db_properties = get_db_properties()
        db_ratings_2019 = spark.read.jdbc(
            url=db_properties["url"], table="filtered_ratings_2019", properties=db_properties
        )
        db_ratings_2019 = db_ratings_2019.select(
            col("user_id").alias("userId"),
            col("movie_id").alias("movieId"),
            col("rating"),
        )
        movies = spark.read.jdbc(
            url=db_properties["url"], table="movies", properties=db_properties
        )
        movies = movies.select(
            col("movie_id").alias("movieId"), col("title"), col("genres")
        )
        movies_ratings_2019 = spark.read.jdbc(
            url=db_properties["url"],
            table="movies_ratings_2019",
            properties=db_properties,
        )
        movies_ratings_2019 = movies_ratings_2019.select(
            col("movie_id").alias("movieId"),
            col("title"),
            col("genres"),
            col("avg_rating"),
            col("total_ratings"),
        )

        # Prepare ratings dataframe
        movies_user_has_seen = get_user_ratings(self)
        user_movies = dict(zip(self.user_input_ids, self.user_ratings))
        new_rtrue = get_new_rtrue(self, db_ratings_2019)

        # Run the model
        als = ALS(
            rank=20,
            maxIter=15,
            regParam=0.01,
            # implicitPrefs=True,
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            coldStartStrategy="drop",
        )
        model = als.fit(new_rtrue)

        # Filter and reshape recommendations
        recommended_movies = get_recommendations_for_new_user(model)
        top_genre = get_relevant_genre(user_movies, movies)
        filtered_recommended_movies = filter_recommendations(
            recommended_movies, movies_ratings_2019
        )
        recommendations = output_shape(
            filtered_recommended_movies, movies_user_has_seen
        )

        return recommendations


if __name__ == "__main__":
    example_input = {
        "seen1": "True",
        "rating1": "15",
        "seen3": "True",
        "rating3": "32",
        "seen4": "True",
        "rating4": "28",
        "seen5": "True",
        "rating5": "25",
        "seen6": "True",
        "rating6": "25",
        "seen7": "True",
        "rating7": "49",
        "seen8": "True",
        "rating8": "50",
        "seen9": "True",
        "rating9": "34",
        "seen10": "True",
        "rating10": "50",
    }
    spark_recommender = sparkRecommender(example_input)
    print(spark_recommender.als())
    spark.stop()
