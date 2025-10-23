import matplotlib
matplotlib.use("Agg")  #use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from wordcloud import WordCloud
import io
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import pickle
from matplotlib.dates import mdates 