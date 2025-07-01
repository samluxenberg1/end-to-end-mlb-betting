import pandas as pd
from utils_features import merge_team_stats_features_into_games

# Assume we already have df (combined team_stats dataframe) sorted by team and game_date_time
def rolling_avg_team_stat(df: pd.DataFrame, team_stat: str, window: int) -> pd.Series:
    return (
        df
        .groupby(['team_id','season'])[team_stat]
        .rolling(window=window, min_periods=1, closed='left')
        .mean()
        .fillna(value=0)
        .reset_index(drop=True)
    )

def rolling_std_team_stat(df: pd.DataFrame, team_stat: str, window: int) -> pd.Series:
    return (
        df
        .groupby(['team_id','season'])[team_stat]
        .rolling(window=window, min_periods=1, closed='left')
        .std()
        .fillna(value=0)
        .reset_index(drop=True)
    )

def get_rolling_team_stat_features(df: pd.DataFrame, team_stat_cols: list[str], window: list[int] = [7]) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    for stat in team_stat_cols:
        for w in window:
            df[f"rolling_avg_{stat}_{w}days"] = rolling_avg_team_stat(df, team_stat=stat, window=w)
            df[f"rolling_std_{stat}_{w}days"] = rolling_std_team_stat(df, team_stat=stat, window=w)
    
    return df

def lag_team_stat(df: pd.DataFrame, team_stat: str, lag: int) -> pd.Series:
    return (
        df
        .groupby(['team_id','season'])[team_stat]
        .shift(lag)
        .fillna(value=0)
        .reset_index(drop=True)
    )
def get_lag_team_stat_features(df: pd.DataFrame, team_stat_cols: list[str], lags: list[int]) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    for stat in team_stat_cols:
        for lag in lags:
            df[f"{stat}_lag{lag}"] = lag_team_stat(df, team_stat=stat, lag=lag)

    return df

def create_team_stats_features(df_games: pd.DataFrame, df_team_stats: pd.DataFrame, window: list[int], lags: list[int]) -> pd.DataFrame:
    team_stat_cols = ['obp','slg','era','whip','strikeouts_pitching','baseonballs_pitching','hits_pitching']
    df_team_stats_rolling = get_rolling_team_stat_features(
        df_team_stats, 
        team_stat_cols=team_stat_cols, 
        window=window
    )
    rolling_avg_team_stat_cols = [f"rolling_avg_{stat}_{w}days" for stat in team_stat_cols for w in window]
    rolling_std_team_stat_cols = [f"rolling_std_{stat}_{w}days" for stat in team_stat_cols for w in window]
    rolling_team_stat_cols = rolling_avg_team_stat_cols + rolling_std_team_stat_cols

    df_team_stats_rolling_lags = get_lag_team_stat_features(
        df_team_stats_rolling,
        team_stat_cols=team_stat_cols,
        lags=lags
    )
    lags_team_stat_cols = [f"{stat}_lag{lag}" for stat in team_stat_cols for lag in lags]
    rolling_lags_team_stat_cols = rolling_team_stat_cols + lags_team_stat_cols
    
    df_games = merge_team_stats_features_into_games(
        df_games=df_games, 
        df_team_stats=df_team_stats_rolling_lags,
        team_stats_cols=rolling_lags_team_stat_cols
    )

    # Add home - away (or home / away) for rolling various features (not lagged features)
    for col in rolling_team_stat_cols:
        df_games[f"{col}_diff"] = df_games[f"home_{col}"] - df_games[f"away_{col}"]
        df_games[f"{col}_ratio"] = df_games[f"home_{col}"] / (df_games[f"away_{col}"] + 1e-8) 

    return df_games