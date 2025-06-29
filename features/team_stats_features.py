import pandas as pd
from utils_features import merge_team_stats_features_into_games

# Assume we already have df_ts_c (combined team_stats dataframe)
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

def create_team_stats_features(df_games: pd.DataFrame, df_team_stats: pd.DataFrame, window: list[int]) -> pd.DataFrame:
    team_stat_cols = ['obp','slg','era','whip','strikeouts_pitching','baseonballs_pitching','hits_pitching']
    df_team_stats_rolling = get_rolling_team_stat_features(
        df_team_stats, team_stat_cols=team_stat_cols, 
        window=window
    )
    rolling_avg_team_stat_cols = [f"rolling_avg_{stat}_{w}days" for stat in team_stat_cols for w in window]
    rolling_std_team_stat_cols = [f"rolling_std_{stat}_{w}days" for stat in team_stat_cols for w in window]
    rolling_team_stat_cols = rolling_avg_team_stat_cols + rolling_std_team_stat_cols

    print(df_team_stats.head())
    df_games = merge_team_stats_features_into_games(
        df_games=df_games, 
        df_team_stats=df_team_stats_rolling,
        team_stats_cols=rolling_team_stat_cols
    )

    # add home - away (or home / away) for various features

    return df_games