import pandas as pd

from load_games_from_db import load_games_from_db
from load_team_stats_from_db import load_team_stats_from_db
from games_features import create_game_features
from utils_features import merge_games_identifiers_into_team_stats
from team_stats_features import create_team_stats_features
"""
Plan: 

Develop individual types of features in their own separate scripts (games features, team stats features, player stats features)
and call those feature engineering functions together here in one build_features() function. 
"""

def build_features():

    # Step 1: Load games data
    df_games = load_games_from_db()

    # Step 2: Load team stats data
    df_team_stats = load_team_stats_from_db()

    # Step 3: Load player stats data

    # Step 4: Feature engineering
    # Convert to datetime
    datetime_cols = ['game_date','game_date_time']
    for col in datetime_cols:
        df_games[col] = pd.to_datetime(df_games[col])
        
    #df_games['game_date'] = pd.to_datetime(df_games['game_date'])
    #df_games['game_date_time'] = pd.to_datetime(df_games['game_date_time'])

    # Rename teams whose names changed
    df_games.loc[df_games['home_team']=='Cleveland Indians', 'home_team'] = 'Cleveland Guardians'
    df_games.loc[df_games['away_team']=='Cleveland Indians', 'away_team'] = 'Cleveland Guardians'

    df_games.loc[df_games['home_team']=='Athletics', 'home_team'] = 'Oakland Athletics'
    df_games.loc[df_games['away_team']=='Athletics', 'away_team'] = 'Oakland Athletics'

    ## Feature engineering on games 
    df_games = create_game_features(df_games, date_col='game_date',date_time_col='game_date_time')

    ## Feature engineering on team stats
    # Merge game identifiers into team stats
    df_team_stats = merge_games_identifiers_into_team_stats(df_games=df_games, df_team_stats=df_team_stats)
    for col in datetime_cols:
        df_team_stats[col] = pd.to_datetime(df_team_stats[col])
        
    df_games = create_team_stats_features(df_games=df_games, df_team_stats=df_team_stats, window=[7,14], lags=[1,2,3])

    ## Feature engineering on player stats

    # Step 2: Define target
    df_games['home_win'] = (df_games['home_score'] > df_games['away_score']).astype(int)

    
    
    
    # Date related features
    

    # Step 4: Drop any columns not needed for modeling
    #df_model = df.drop(columns=['xxx','yyy'])
    
    return df_games#_model

if __name__=='__main__':
    
    output_csv = 'data/processed/model_data.csv'
    df_model = build_features()
    d = df_model
    cols = ['game_id','game_date_time','home_team','away_team','home_score','away_score','home_rolling_avg_obp_7days','rolling_avg_obp_7days_diff', 'home_obp_lag1','home_obp_lag2','home_obp_lag3']
    pd.set_option('display.max_columns', None)
    #print(d.info())
    #print(d[cols])
    print("\nNew York Yankees")
    print(d.query("home_team=='New York Yankees' | away_team=='New York Yankees'")[cols].head())
    print(d.columns[:30])
    #df_model.to_csv(output_csv, index=False)
    #print(f"Saved {len(df_model)} rows and {len(df_model.columns)} columns to {output_csv}")




