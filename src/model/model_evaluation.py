import pandas as pd
import numpy as np
import os
import yaml
import pickle
import logging 
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature
from sklearn.metrics import classification_report, confusion_matrix
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
        logger.error(f"failed to poarse the csv file: {e}")
    except Exception as e:
        logger.error(f"unexpecteed error occur while loadimg the data from {file_path}: {e}")
        raise

def load_model(model_path: str) ->None:
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.debug(f"model successfully .loaded from: {model_path}")
        return model
    except Exception as e:
        logger.error(f"unexpected error occur while loading data from {model_path}: {e}")

def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    try:
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        logger.debug(f"TfidfVectorizer loaded successfully from: {vectorizer_path}")
        return vectorizer
    except Exception as e:
        logger.error(f"unexpected error occer while loading TfidfVectorizer: {e}")
        raise

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
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

def evaluate_model(model, x_test:np.ndarray, y_test: np.ndarray):
    try:
        y_pred = model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict = True)
        cm = confusion_matrix(y_test, y_pred)
        logger.debug("Model evaluation completed")
        return report, cm
    except Exception as e:
        logger.error("Error occur during modek evaluation: %s", e)

def log_confusion_matrix(cm, dataset_name: str):
    plt.figure(figsize = (8,6))
    sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues")
    plt.title(f"Confussion Matrix for {dataset_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # save the confussion matrix plot and log it to mlflow
    cm_file_path = f"confussion_matrix_{dataset_name}.png"
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        # create a dictionary with the info you want to save
        model_info = {
            "run_id": run_id,
            "model_path": model_path
        }

        # save the dictionary as a json file
        with open(file_path, "w") as f:
            json.dump(model_info, f, indent = 4)
        logger.debug("Model info saved to %s", file_path)
    except Exception as e:
        logger.error("Error occur while saving model info: %s", e)
        raise

def get_root_directory() ->str:
    """get the root directory (two level up from the script location)"""
    try:
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(cur_dir, "../../"))
    except Exception as e:
        logger.error("Unexpected error occur when getting the root directory: %s", e)

def main():

    # mlflow.set_tracking_uri("http://ec2-13-217-3-143.compute-1.amazonaws.com:5000")
    mlflow.set_experiment("Demo_run")
    with mlflow.start_run() as run:
        try:
            # load parameter from yaml file
            root_dir = get_root_directory()
            params = load_params(params_path = os.path.join(root_dir, "params.yaml"))
            print(type(params))

            # log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # load model and vectorizer
            model = load_model(model_path = os.path.join(root_dir, "model/model.pkl"))
            vectorizer = load_vectorizer(vectorizer_path = os.path.join(root_dir, "tfidf_vectorizer.pkl"))

            # load test_data
            test_df = load_data(os.path.join(root_dir, "data/preprocessed/test_preprocessed.csv"))
            # prepare test_data
            x_test = vectorizer.transform(test_df["clean_comment"].values)
            y_test = test_df["category"].values

            # create dataframe for signature inference (using the first few rows as an example
            input_example = pd.DataFrame(x_test.toarray()[:5],columns = vectorizer.get_feature_names_out()) #added for input example
            # infer the signature
            signature = infer_signature(input_example, model.predict(x_test[:5])) #added for signature

            # log model with signature
            mlflow.sklearn.log_model(
                model, "lgbm_model", signature = signature, input_example = input_example
            )

            # save the model info
            # artifact_uri = mlflow.get_artifact_uri()
            model_path = "lgbm_model"
            save_model_info(run_id = run.info.run_id, model_path = model_path, file_path = os.path.join(root_dir,"experiment_info.json"))

            # log the vectorizer as an artifact
            mlflow.log_artifact(os.path.join(root_dir, "tfidf_vectorizer.pkl"))

            # evaluate the model and get metrics
            report, cm = evaluate_model(model = model, x_test = x_test, y_test = y_test)

            # log classification and metrics for the test data
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics["precision"],
                        f"test_{label}_recall": metrics["recall"],
                        f"test_{label}_f1-score": metrics["f1-score"]
                    })
                    # for metric, value in metrics.items():
                    #     mlflow.log_metric(f"test_{label}_{metric}", value)
            
            # log confusion matrix
            log_confusion_matrix(cm, "Test Data")

            # add important tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTbe Comment")

        except Exception as e:
            logger.error(f"failed to complete model evaluation: {e}")
            raise

if __name__ == "__main__":
    main()