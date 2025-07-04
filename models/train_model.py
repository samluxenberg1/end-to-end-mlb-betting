import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, log_loss 

from utils_models import load_model_data

input_model_data = "data/processed/model_data.csv"
df = load_model_data(input_model_data)

assert 'home_win' in df.columns, "Target column 'home_win' not found"