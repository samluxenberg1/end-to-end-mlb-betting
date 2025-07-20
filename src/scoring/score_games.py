import argparse
import logging
import pandas as pd
import joblib
import datetime
from datetime import datetime, date, timedelta

import os
import psycopg2

from src.features.games_features import create_game_features
from src.features.team_stats_features import create_team_stats_features
#from features.player_stats_features import create_player_stats_features
#from src.features import load_games_from_db # need to add load_games_for_date(game_date)
#from features import load_team_stats_from_db
from src.features import build_features

from src.etl.utils import fetch_games
from src.scoring import load_games_from_db_scoring
from src.models.train_model import MLBModelTrainer
from src.models.model_classes import ModelConfig, TrainingResults
from src.gen_utils import load_config
"""
For scoring, there are 5 steps to accomplish. 

    1. Fetch today's games
    2. Fetch associated team/player stats
    3. Generate features using your existing feature scripts
    4. Load trained model (joblib.load("models/baseline_model.pk"))
    5. Apply model and output predictions
"""

DB_CONFIG = {
        "dbname": os.environ['DB_NAME'],
        "user": os.environ['DB_USER'],
        "password": os.environ['DB_PASSWORD'],
        "host": os.environ['DB_HOST'],
        "port": os.environ['DB_PORT']
    }
MODEL_VERSION = "v1.0"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def insert_predictions_to_db(df: pd.DataFrame, table_name: str = "predictions_log"):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    insert_query = f"""
        INSERT INTO {table_name} (
            game_id, game_date, home_team, away_team, predicted_prob_home_win, prediction_time, model_version
        ) VALUE (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (game_id, model_version) DO NOTHING;
    """
    now = datetime.utcnow()

    for _, row in df.iterrow():
        cur.execute(
            insert_query,
            (
                row['game_id'],
                row['game_date'],
                row['home_team'],
                row['away_team'],
                row['predicted_prob_home_win'],
                now, 
                MODEL_VERSION
            )
        )
    conn.commit()
    cur.close()
    conn.close()


def main(scoring_date: str): 
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Scoring games for {scoring_date}")

    # Step 1: Fetch games that are scheduled for today and tomorrow
    #scoring_games_df = load_games_from_db_scoring.load_games_from_db()
    #scoring_games_df = scoring_games_df[scoring_games_df['game_date']>=date.today()].sort_values('game_date_time')

    # Step 2: Fetch all historical games
    #historical_games_df = load_games_from_db.load_games_from_db()

    # Step 2: Fetch all historical team stats
    #team_stats_df = load_team_stats_from_db.load_team_stats_from_db()

    # Step 2: Fetch all historical player stats
    #player_stats_df = load_player_stats_from_db()

    # Step 3: Only keep team stats from games in games data

    
    # Step 2: Build features
    features_df = build_features.build_features()
    features_df.loc[features_df['state']=='Scheduled', 'home_win'] = pd.NA
    # Prep scoring data
    # Remove columns
    cols_to_remove = [
            "game_id", "game_date", "game_date_time", "home_team_id",
            "away_team_id", "home_score", "away_score", "state", "home_win"
        ]
    X_scoring = features_df.drop(cols_to_remove, axis=1)
    

    # Step 3: Load model
    # config = ModelConfig(
    #         model_type="catboost",
    #         hyperparams={
    #             #'objective': 'binary',
    #             'loss_function': 'Logloss',
    #             'verbose': False,
    #             # 'n_estimators': 500,        # Increased estimators for LGBM
    #             # 'learning_rate': 0.05,      # Learning rate for boosting
    #             # 'num_leaves': 31,           # Controls complexity of trees
    #             # 'max_depth': -1,            # No limit on depth, often works well with num_leaves
    #             # 'min_child_samples': 20,    # Minimum data in a leaf
    #             # 'subsample': 0.8,           # Fraction of samples to be randomly sampled
    #             # 'colsample_bytree': 0.8,    # Fraction of features to be randomly sampled per tree
    #             # 'random_state': 888,
    #             # 'n_jobs': -1,
    #             # 'reg_alpha': 0.1,           # L1 regularization
    #             # 'reg_lambda': 0.1          # L2 regularization
    #         },
    #         cv_params={
    #             'test_size': 30,
    #             #'train_size': 360,
    #             'n_splits': 3, # for testing purposes
    #             'gap_size': 3,
    #             'window_type': 'rolling'
    #         },
    #         #categorical_cols=['away_team'], 
    #         categorical_cols=['home_team','away_team','venue','game_type'],
    #         encoding_type='category',
    #         calibration_method='isotonic',
    #         random_state=888
    #     )
    model = MLBModelTrainer.load_models("src/saved_models/trained_mlb_model.pkl")

    # Step 4: Make predictions on scheduled games
    scoring_classifications = model.predict(X_scoring)
    scoring_probabilities = model.predict_proba(X_scoring)[:, 1]
    
    # Step 5: Attache results to scoring data
    features_df['scoring_classifications'] = scoring_classifications
    features_df['scoring_probabilities'] = scoring_probabilities
    #features_df['predicted_prob_home_win'] = predict_proba(model, features_df)

    # Step 5: Log to DB
    #insert_predictions_to_db(features_df)
    #logging.info("Predictions saved to database.")

    return features_df



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=date.today().isoformat())
    args = parser.parse_args()
    feat_df = main(args.date)
    pd.set_option('display.max_columns', None)
    config = load_config('config.yml')
    #print(feat_df.tail(2))
    print(f"Config...\n")
    print(f"Target: {config['target']}")
    print(f"Columns to Remove: {config['cols_to_remove']}")
    print(f"Config: {config}")
    print("\n")
    print(f"feat_df shape: {feat_df.shape}")
    print(feat_df['game_date'].min())
    print(feat_df['game_date'].max())
    print(feat_df['game_date_time'].max())
    feat_df.to_csv("src/scoring/output.csv", index=False)