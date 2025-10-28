
import yaml
import os
from dotenv import load_dotenv

# def load_params(params_path: str) -> dict:
#     with open(params_path, "r") as f:
#         params = yaml.safe_load(f)
#     return params


# def get_root_directory() ->str:
#     """get the root directory (two level up from the script location)"""
#     cur_dir = os.path.dirname(os.path.abspath(__file__))
#     return os.path.abspath(os.path.join(cur_dir, "../../"))

# # load parameter from yaml file
# root_dir = get_root_directory()
# params = load_params(params_path = os.path.join(root_dir, "params.yaml"))
# print(type(params))

# # log parameters
# for key, value in params.items():
#     print(key, value)

load_dotenv()
def get_api_key():
    API_key = os.getenv("API_key")
    return {"api_key":API_key}

api = get_api_key()["api_key"]

print(api)