import pandas as pd

from load_games_from_db import load_games_from_db
from games_features import create_game_features
"""
Plan: 

Develop individual types of features in their own separate scripts (games features, team stats features, player stats features)
and call those feature engineering functions together here in one build_features() function. 
"""

def build_features():

    # Step 1: Load games data
    df = load_games_from_db()

    

    # Step 2: Load team stats data
    
    # Step 3: Load player stats data

    # Step 4: Feature engineering
    # Convert to datetime
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['game_date_time'] = pd.to_datetime(df['game_date_time'])

    # Rename teams whose names changed
    df.loc[df['home_team']=='Cleveland Indians', 'home_team'] = 'Cleveland Guardians'
    df.loc[df['away_team']=='Cleveland Indians', 'away_team'] = 'Cleveland Guardians'

    df.loc[df['home_team']=='Athletics', 'home_team'] = 'Oakland Athletics'
    df.loc[df['away_team']=='Athletics', 'away_team'] = 'Oakland Athletics'

    ## Feature engineering on games 
    df = create_game_features(df, date_col='game_date',date_time_col='game_date_time')

    ## Feature engineering on team stats

    ## Feature engineering on player stats

    # Step 2: Define target
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)

    
    
    
    # Date related features
    

    # Step 4: Drop any columns not needed for modeling
    #df_model = df.drop(columns=['xxx','yyy'])
    
    return df#_model

if __name__=='__main__':
    
    output_csv = 'data/processed/model_data.csv'
    df_model = build_features()
    d = df_model.head()
    print(d.head())
    
    #df_model.to_csv(output_csv, index=False)
    #print(f"Saved {len(df_model)} rows and {len(df_model.columns)} columns to {output_csv}")




