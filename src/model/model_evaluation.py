import pandas as pd
import numpy as np
import os
import yaml
import pickle
import logging 
import mlflow
import mlflow.sklearn
import matplotlib.pyplot
import seaborn as sns
import json
from mlflow.models import infer_signature
from sklearn.metrics import classification_report, confusion_matix
from sklearn.feature_extraction.text import TfidfVectorizer


# logging configuration
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_evaluation_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna("", inplace = True)
        logger.debug("data loaded succeassfully")
        return df
    except pd.errors.ParseError as e:
        logger.errord(f"failed to poarse the csv file: {e}")
    except Exception as e:
        logger.error(f"unexpecteed error occur while loadimg the data from {file_path}: {e}")
        raise

def load_model(model_path: str) ->None:
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.debug(f"model succefully .loaded from: {model_path}")
        return model
    except Exception as e:
        logger.error(f"unexpected error occur while loading data from {model_path}: {e}")
        