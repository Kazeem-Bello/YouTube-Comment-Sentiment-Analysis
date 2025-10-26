import matplotlib
matplotlib.use("Agg")  #use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from wordcloud import WordCloud
import io
import os
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import pickle
import matplotlib.dates as mdates 

app = Flask(__name__)
CORS(app)  #Enable CORS for all routes

# define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformation to a comment"""
    try:
        # convert to lowercase
        comment = comment.lower()

        # remove trailing and leading whitespaces
        comment = comment.strip()

        # remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', comment)

        # remove stopwords but retain important words
        stop_words = set(stopwords.words('english')) - {'not', 'no', 'nor', 'but', 'however', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment
    
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        raise


# load the model and vectorizer from the modle registry and local storage 
def load_model_and_vectorizer(model_name: str, model_version: str, vectorizer_path: str):
    try: 
        # mlflow.set_tracking_uri("")
        client = MlflowClient()
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
        with open (vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

        return model, vectorizer
    except Exception as e:
        print(f"Error loading model and vectorizer: {e}")
        raise

# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "11", "./tfidf_vectorizer.pkl")

def load_model_vectorizer(model_path: str, vectorizer_path: str):
    if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
    if not os.path.exists(model_path):
            raise FileNotFoundError(f"model file not found at {model_path}")
    with open (model_path, "rb") as f:
        model = pickle.load(f)
    with open (vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# model, vectorizer= load_model_vectorizer("./model/model.pkl", "./tfidf_vectorizer.pkl")



@app.route("/")
def home():
    return "welcome to our flask api"


@app.route("/predict", methods = ["POST"])
def predict():
    data = request.json
    comments = data.get("comments")
    # print("I am the comment: ", comments)
    # print("I am the comment type:", type(comments))
    # print("I am the comment shape:", len(comments))

    if not comments:
        return jsonify({"error": "No comments provided"}), 400
    try:
        # preprocessed each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # convert the sparse matrix to dense format
        dense_comments = transformed_comments.toarray()


        # make predictions
        input_comments = pd.DataFrame(dense_comments, columns = vectorizer.get_feature_names_out())
        predictions = model.predict(input_comments).tolist()

        # convert prediction to strings for consistency
        # predictions = [str(x) for x in predictions]

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # return the response with original comment ansd predicted sentiments
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)


if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5002, debug = True)
