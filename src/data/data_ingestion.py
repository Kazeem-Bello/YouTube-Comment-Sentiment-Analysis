import pandas as pd
import numpy as np
import os
import yaml
import logging 
from sklearn.model_selection import train_test_split

# logging configuration
logger = logging.getLogger("data_imgestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("errors.log")
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
