import argparse
import logging
import pandas as pd
import joblib
from datetime import datetime

from features.games_features import create_game_features
from features.team_stats_features import create_team_stats_features
#from features.player_stats_features import create_player_stats_features
from features.load_games_from_db import load_games_from_db # need to add load_games_for_date(game_date)
from features.load_team_stats_from_db import load_team_stats_from_db

"""
For scoring, there are 5 steps to accomplish. 

    1. Fetch today's games
    2. Fetch associated team/player stats
    3. Generate features using your existing feature scripts
    4. Load trained model (joblib.load("models/baseline_model.pk"))
    5. Apply model and output predictions
"""

def load_model(model_path: str):
    return joblib.load(model_path)

def prepare_features(game_date: str) -> pd.DataFrame:
    logging.info(f"Preparing features for {game_date}")

    # Load base data
    games_df = load_games_from_db()

def score_games():
    return 

def main():
    return 