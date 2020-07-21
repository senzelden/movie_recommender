from flask import Flask, render_template, request
from recommender import random_recommend, MOVIES
from nmf_recommend import nmf_recommender

app = Flask(__name__)
# __name__ is imply a reference to the current python script / module. All relative paths will be centered around this.

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', choices=MOVIES)

@app.route('/hello/<name>')
def hello(name):
    name = name.upper()
    return render_template('hello.html', name_html=name)

@app.route('/recommendation')
def recommend():

    user_input = dict(request.args)
    recommendations = nmf_recommender()  # at a later point nmf_recommender would take in user_input
    """Here is where you take the user input dictionary and pre-process it and use it for the recommendation f"""


    movies = random_recommend(MOVIES, 5)
    return render_template('recommendation.html', movies=recommendations, input=user_input)

@app.route('/recommendation_test')
def recommend_test():
    user_rating = dict(request.args)
    return render_template('recommendation_test.html', user_rating=user_rating)


if __name__ == '__main__':
    # if i run "python application.py", please run the following code
    app.run(debug=True, use_reloader=True)
