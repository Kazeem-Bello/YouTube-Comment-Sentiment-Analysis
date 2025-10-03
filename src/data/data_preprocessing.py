import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging


# logging configuration
logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("prprocessing_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# download the required NLTK data
nltk.download("wordnet")
nltk.download("stopwords")

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
        logger.error(f"Error in preprocessing comment: {e}")
        raise


def normalize_text(df:pd.DataFrame)-> pd.DataFrame:
    """Applying preprocessing to the text data in the dataframe"""
    try:
        df["clean_comment"] = df["clean_comment"].apply(preprocess_comment)
        logger.debug(f"{df} normalization completed")
        return df
    
    except Exception as e:
        logger.error("Error during text normalization: %s", e)
        raise


def save_data(train_data:pd.DataFrame, test_data:pd.DataFrame, data_path:str)-> None:
    try:
        preprocessed_data_path = os.path.join(data_path, "preprocessed")
        logger.debug("Creating directory %s", preprocessed_data_path)

        os.makedirs(preprocessed_data_path, exist_ok = True)
        logger.debug("Directory %s created or already exists", preprocessed_data_path)

        train_data.to_csv(os.path.join(preprocessed_data_path, "train_preprocessed.csv"), index = False)
        test_data.to_csv(os.path.join(preprocessed_data_path, "test_preprocessed.csv"), index = False)
        logger.debug("Preproccesed data saved to %s", preprocessed_data_path)

    except Exception as e:
        logger.error("Error occured while saving data: %s", e)
        raise

def main():
    try:
        logger.debug(f"starting data preprocessing...")

        # fetch the data from data/raw
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug("Data loaded successfully")

        train_preprocessed_data = normalize_text(train_data)
        test_preprocessed_data = normalize_text(test_data)

        save_data(train_data = train_preprocessed_data, test_data = test_preprocessed_data, data_path = "././data")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()



        