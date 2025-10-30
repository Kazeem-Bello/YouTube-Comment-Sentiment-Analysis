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
from dotenv import load_dotenv


load_dotenv()

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


# load the model and vectorizer from the model registry 
# def load_model_and_vectorizer(model_name: str, model_version: str, vectorizer_path: str):
#     try: 
#         # mlflow.set_tracking_uri("")
#         client = MlflowClient()
#         model_uri = f"models:/{model_name}/{model_version}"
#         model = mlflow.pyfunc.load_model(model_uri)
#         if not os.path.exists(vectorizer_path):
#             raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
#         with open (vectorizer_path, "rb") as f:
#             vectorizer = pickle.load(f)

#         return model, vectorizer
#     except Exception as e:
#         print(f"Error loading model and vectorizer from mlflow: {e}")
#         raise

# # Initialize the model and vectorizer
# model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "11", "./tfidf_vectorizer.pkl")


# load the model and vectorizer from local storage 
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

model, vectorizer= load_model_vectorizer("model/model.pkl", "tfidf_vectorizer.pkl")

@app.route("/get_api_key")
def get_api_key():
    API_key = os.getenv("API_key")
    return jsonify({"api_key":API_key})


@app.route("/")
def home():
    return "welcome to our flask api"


@app.route("/predict_with_timestamp", methods = ["POST"])
def predict_with_timestamp():
    data = request.json
    comment = data.get("comments")

    if not comment:
        return jsonify({"error": "No comments provided"}), 400
    try:
        comments = [item["text"] for item in comment]
        timestamps = [item["timestamp"] for item in comment]

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
    
    # return the response with original comment ansd predicted sentiments and timestamp
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)


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


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png') 
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500






if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8080, debug = True)
