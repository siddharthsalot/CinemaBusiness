from flask import Flask, flash, current_app, request
from flask import render_template, url_for
from wtforms import SubmitField, StringField
from flask_wtf import FlaskForm
import os
import secrets
from sys import stderr
import numpy as np
from wtforms.validators import DataRequired
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

app.config['SECRET_KEY'] = '1d7ce38b6bfe845340bbd1ac902cbe4f'
collection_dict = {0: "Low", 1: "Medium", 2: "High"}


class SentimentAnalyzerForm(FlaskForm):
    imdb_rating = StringField(
        'IMDB Rating', validators=[DataRequired()])
    sentiment_score = StringField(
        'Sentiment Score', validators=[DataRequired()])
    theatres = StringField(
        'Theatres', validators=[DataRequired()])
    submit = SubmitField("Run")


def classify(input_data):
    classifier = joblib.load('CinemaBusiness.pkl')
    sc_X = joblib.load('Feature_Scaling.pkl')
    X = sc_X.transform(input_data)
    val = classifier.predict(X)
    return collection_dict[val[0]]


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def test_img():
    form = SentimentAnalyzerForm()
    if request.method == "POST":
        imdb_rating = form.imdb_rating.data
        sentiment_score = form.sentiment_score.data
        theatres = form.theatres.data
        input_data = [[imdb_rating, sentiment_score, theatres]]
        output = classify(input_data)
        if(output == "High"):
            msg_color = "success"
        if(output == "Medium"):
            msg_color = "warning"
        if(output == "Low"):
            msg_color = "danger"
        flash(f"Our prediction is that this movie's opening collection would be: " + output, msg_color)
        return render_template("home.html", form=form, legend="Predict Opening Collection", title="Predict Opening Collection")
    else:
        return render_template("home.html", form=form, legend="Predict Opening Collection", title="Predict Opening Collection")


if __name__ == "__main__":
    app.run(debug=True)
