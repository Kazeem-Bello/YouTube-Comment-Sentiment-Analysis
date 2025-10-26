import json
import mlflow
import logging
import os

# logging configuration
logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_registration_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# mlflow.set_tracking_uri("http://ec2-13-217-3-143.compute-1.amazonaws.com:5000")
def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file"""
    try: 
        with open(file_path, "r") as f:
            model_info = json.load(f)
        logger.debug("model info loade from %s", file_path)
        return model_info
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading the model in: %s", e)
        raise

def register_model(model_name: str, model_info: dict):
    "Register the model to mlflow model registry"
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        # model_uri = model_info['model_path']
        model_version = mlflow.register_model(model_uri, model_name) #register the model
        # transition the model to stagging area
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name = model_name,
            version = model_version.version,
            stage = "staging"
        )
        logger.debug(f"Model {model_name} version {model_version.version} registered and transition to staging")
    except Exception as e:
        logger.error("Error during model registration, %s", e)
        raise

def main():
    try:
        model_info_path = "./experiment_info.json"
        model_info = load_model_info(model_info_path)

        model_name = "yt_chrome_plugin_model"
        register_model(model_name, model_info)
        logger.debug("Model successfully registered")
    except Exception as e:
        logger.error("failed to complete the model registration process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()