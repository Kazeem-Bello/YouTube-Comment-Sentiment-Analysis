import pandas as pd
import numpy as np
import os
import yaml
import logging 
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import pickle 
from sklearn.feature_extraction.text import TfidfVectorizer

# logging configuration
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_building_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters frfom a yaml file"""
    try: 
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("parameters retrieved from %s", params_path)
        return params
    
    except FileNotFoundError:
        logger.error("File not Found: %s", params_path)
        raise
    except  yaml.YAMLError:
        logger.error("YAML error: %s", params_path)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise


def load_data(file_path: str)-> pd.DataFrame:
    """load data from a csv file"""
    try:
        df = pd.read_csv(file_path)
        df.fillna("", inplace = True) #fill any NaN values
        logger.debug("Data loaded and Nan filled from %s", file_path)
        return df       
    except pd.errors.ParserError as e:
        logger.error("failed to parse the csv file: %s", e)
        raise
    except Exception as e:
        logger.error("unexpected error occur while loading the data, %s", e)
        raise

def apply_tfidf(train_data:pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    """Apply Tf-IDF with ngrams to the data"""
    try:
        vectorizer = TfidfVectorizer(max_features = max_features, ngram_range = ngram_range)
        x_train = train_data["clean_comment"].values
        y_train = train_data["category"].values

        # perform Tf-idf vectorizer
        x_train_tfidf = vectorizer.fit_transform(x_train)
        logger.debug("TF-idf transformation complete, train shape: %s", x_train_tfidf.shape)

        # save the vectorizer in the root directory
        with open(os.path.join(get_root_directory(), "tfidf_vectorizer.pkl"), "wb") as f:
            pickle.dump(vectorizer, f)

        logger.debug("TF-IDF applied with trigrams and data transformed")
        return x_train_tfidf, y_train
    
    except Exception as e:
        logger.error("Error during TF-IDF transformation: %s", e)
        raise


def train_lgm(x_train:np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth:int, n_estimators: int) -> lgb.LGBMClassifier:
    """Train a lightGBM model"""
    try:
        best_model = lgb.LGBMClassifier(
            objective = "multiclass",
            num_class = 3,
            metric = "multi_logloss",
            is_unbalance = True,
            class_weight = "balanced",
            reg_alpha = 0.1,
            reg_lambda = 0.1,
            learning_rate = learning_rate,
            max_depth = max_depth,
            n_estimators = n_estimators
        )
        best_model.fit(x_train, y_train)
        logger.debug("LightGBM model training completed")
        return best_model
    except Exception as e:
        logger.error("Error occur during LightGBM model training: %s", e)
        raise

def save_model(model, file_path: str) -> None:
    """save the model to a file"""
    try:
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logger.debug("Model saved to %s", file_path)
    except Exception as e:
        logger.error("Error occur while saving the model", e)
        raise


def get_root_directory() ->str:
    """get the root directory (two level up from the script location)"""
    try:
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(cur_dir, "../../"))
    except Exception as e:
        logger.error("Unexpected error occur when getting the root directory: %s", e)


def main():
    try:
        # get root directory and resolve the path for params.yaml
        root_dir = get_root_directory()

        # load parameters from the root directory
        params = load_params(os.path.join(root_dir, "params.yaml"))
        max_features = params["model_building"]["max_features"]
        ngram_range = tuple(params["model_building"]["ngram_range"])
        learning_rate = params["model_building"]["learning_rate"]
        max_depth = params["model_building"]["max_depth"]
        n_estimators = params["model_building"]["n_estimators"]

        train_df = load_data(os.path.join(root_dir, "data/preprocessed/test_preprocessed.csv"))
        x_train, y_train = apply_tfidf(train_data = train_df, max_features = max_features, ngram_range = ngram_range)
        best_model = train_lgm(x_train = x_train, y_train = y_train, learning_rate = learning_rate, max_depth = max_depth, n_estimators = n_estimators)
        path = os.path.join(root_dir, "model")
        os.makedirs(path, exist_ok = True)
        save_model(model = best_model, file_path = os.path.join(path, "model.pkl"))
    except Exception as e:
        logger.error(f"failed to complete model building: {e}")

if __name__ == "__main__":
    main()



                                                                          
