# YouTube-Comment-Sentiment-Analysis
## MLflow-Basic-Demo



## 1. For Dagshub:

MLFLOW_TRACKING_URI=https://dagshub.com/entbappy/MLflow-Basic-Demo.mlflow \
MLFLOW_TRACKING_USERNAME=entbappy \
MLFLOW_TRACKING_PASSWORD=6824692c47a369aa6f9eac5b10041d5c8edbcef0 \
python script.py


## 2. Exporting variables
```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/entbappy/MLflow-Basic-Demo.mlflow

export MLFLOW_TRACKING_USERNAME=kazeem-bello 

export MLFLOW_TRACKING_PASSWORD=6824692c47a369aa6f9eac5b10041d5c8edbcef0


```


## 3. MLflow on AWS


1. Login to AWS console.
2. Create IAM user with AdministratorAccess
3. Export the credentials in your AWS CLI by running "aws configure"
4. Create a s3 bucket
5. Create EC2 machine (Ubuntu) & add Security groups 5000 port

Run the following command on EC2 machine
```bash
sudo apt update

sudo apt install python3-pip

sudo apt install pipenv

sudo apt install virtualenv

mkdir mlflow

cd mlflow

pipenv install mlflow

pipenv install awscli

pipenv install boto3

pipenv shell


## Then set aws credentials
aws configure


#Finally 
mlflow server -h 0.0.0.0 --default-artifact-root s3://mlflow-test-23  

#open Public IPv4 DNS to the port 5000


#set uri in your local terminal and in your code 
export MLFLOW_TRACKING_URI=http://ec2-54-198-108-56.compute-1.amazonaws.com:5000
```
## 4. Creating and activating a virtual environment
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows



## 5. Here's a **plug-and-play, production-ready logging setup** you can drop into any Python project to manage logs across multiple modules.

---

### ✅ Folder Structure Example

```
your_project/
│
├── logging_config.py         # Sets up the global logging system
├── main.py                   # Your main script
├── module_a.py               # Example module
├── module_b/
│   └── sub_module.py         # Submodule example
└── logs/
    └── errors.log            # Log file will be created here
```

---

## 🧱 Step 1: `logging_config.py`

```python
# logging_config.py

import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging():
    # Create log directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Format for all handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console Handler (DEBUG+)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # Rotating File Handler (ERROR+)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "errors.log"),
        maxBytes=5_000_000,
        backupCount=3
    )
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
```

---

## 🚀 Step 2: `main.py` – Entry Point

```python
# main.py

from logging_config import setup_logging

# Set up logging once here
setup_logging()

import module_a
from module_b import sub_module

logger = logging.getLogger(__name__)
logger.info("Main script started")

module_a.run()
sub_module.do_work()
```

---

## 📦 Step 3: `module_a.py`

```python
# module_a.py

import logging

logger = logging.getLogger(__name__)

def run():
    logger.debug("Running module A")
    try:
        1 / 0
    except ZeroDivisionError:
        logger.error("Division by zero in module A", exc_info=True)
```

---

## 🧩 Step 4: `module_b/sub_module.py`

```python
# module_b/sub_module.py

import logging

logger = logging.getLogger(__name__)

def do_work():
    logger.info("Sub-module B is working")
```

---

## ✅ Result

### Terminal Output (from `console_handler`):

```text
2025-08-29 12:10:01,123 - __main__ - INFO - Main script started
2025-08-29 12:10:01,125 - module_a - DEBUG - Running module A
2025-08-29 12:10:01,127 - module_b.sub_module - INFO - Sub-module B is working
```

### logs/errors.log (from `file_handler`):

```text
2025-08-29 12:10:01,128 - module_a - ERROR - Division by zero in module A
Traceback (most recent call last):
  File "module_a.py", line 7, in run
    1 / 0
ZeroDivisionError: division by zero
```

---

## 💡 Notes

* All modules **use the same logger config**, no duplication.
* You can log `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL` levels.
* Only `ERROR` and above are written to file (you can adjust this).
* Rotating logs prevent massive files.
* You can also extend it to log to external systems (e.g. Slack, Sentry, or ELK stack) later.

---