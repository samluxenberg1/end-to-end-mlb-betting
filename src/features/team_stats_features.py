import pandas as pd
from src.features.utils_features import merge_team_stats_features_into_games

# Assume we already have df (combined team_stats dataframe) sorted by team and game_date_time
def rolling_avg_team_stat(df: pd.DataFrame, team_stat: str, window: int) -> pd.Series:
    """
    Calculate rolling average for a team statistic over a specified window.

    Parameters
    ----------
    df : pd.dataFrame
        DataFrame containing team stats with 'team_id', 'season', and team statistics columns.
    team_stat : str
        Name of team statistical column to calculate rolling average for.
    window : int
        Number of games to include in rolling window
    
    Returns
    -------
    pd.Series
        Series with rolling averages, using left-closed window (excludes current game).

    Note
        Uses closed='left' to prevent data leakage by excluding the current game.
        Assumes df is already sorted by team_id and game_date_time.
    """
    return (
        df
        .groupby(['team_id','season'])[team_stat]
        .rolling(window=window, min_periods=1, closed='left')
        .mean()
        .fillna(value=0)
        .reset_index(drop=True)
    )

def rolling_std_team_stat(df: pd.DataFrame, team_stat: str, window: int) -> pd.Series:
    """
    Calculate rolling standard deviation for a team statistic over a specified window.

    Parameters
    ----------
    df : pd.dataFrame
        DataFrame containing team stats with 'team_id', 'season', and team statistics columns.
    team_stat : str
        Name of team statistical column to calculate rolling standard deviation for.
    window : int
        Number of games to include in rolling window
    
    Returns
    -------
    pd.Series
        Series with rolling standard deviation, using left-closed window (excludes current game).

    Note
        Uses closed='left' to prevent data leakage by excluding the current game.
        Assumes df is already sorted by team_id and game_date_time.
    """
    return (
        df
        .groupby(['team_id','season'])[team_stat]
        .rolling(window=window, min_periods=1, closed='left')
        .std()
        .fillna(value=0)
        .reset_index(drop=True)
    )

def get_rolling_team_stat_features(df: pd.DataFrame, team_stat_cols: list[str], window: list[int] = [7]) -> pd.DataFrame:
    """
    Generate rolling average and standard deviation features for multiple team statistics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing team statistical columns.
    team_stat_cols : list[str]
        List of column names for team statistics to process.
    window : list[int]
        List of window sizes (in games) for rolling calculations. Defaults to [7].

    Returns
    -------
    pd.DataFrame
        DataFrame with original data plus new rolling feature columns
    """
    df = df.copy().reset_index(drop=True)

    for stat in team_stat_cols:
        for w in window:
            df[f"rolling_avg_{stat}_{w}days"] = rolling_avg_team_stat(df, team_stat=stat, window=w)
            df[f"rolling_std_{stat}_{w}days"] = rolling_std_team_stat(df, team_stat=stat, window=w)
    
    return df

def lag_team_stat(df: pd.DataFrame, team_stat: str, lag: int) -> pd.Series:
    """
    Create lagged version of a team statistic.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing team statistics.
    team_stat : str
        Name of statistical column.
    lag : int
        Number of games to shift backward (lag=1 means previous game's value)

    Returns
    -------
    pd.Series
        Series with lagged values, grouped by team_id and season. 

    Note
        Missing values (start of season/team) are filled with 0.
    """
    return (
        df
        .groupby(['team_id','season'])[team_stat]
        .shift(lag)
        .fillna(value=0)
        .reset_index(drop=True)
    )
def get_lag_team_stat_features(df: pd.DataFrame, team_stat_cols: list[str], lags: list[int]) -> pd.DataFrame:
    """
    Generate lagged features for multiple team statistics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing team statistics.
    team_stat_cols : list[str]
        List of column names for team statistics to process.
    lags : list[int]
        List of lag values (number of games to look back).

    Returns
    -------
    pd.DataFrame
        DataFrame with original data plus new lagged feature columns.
    """
    df = df.copy().reset_index(drop=True)

    for stat in team_stat_cols:
        for lag in lags:
            df[f"{stat}_lag{lag}"] = lag_team_stat(df, team_stat=stat, lag=lag)

    return df

def create_team_stats_features(df_games: pd.DataFrame, df_team_stats: pd.DataFrame, window: list[int], lags: list[int]) -> pd.DataFrame:
    """
    Create team statistics features and merge into games data. 

    This function generates rolling averages, rolling standard deviations, 
    and lagged features for key team statistics, then merges them into the 
    games DataFrame with home/away differences and ratios.

    Parameters
    ----------
    df_games : pd.DataFrame
        Input games DataFrame.
    df_team_stats : pd.DataFrame
        DataFrame containing team statistics by game.
    window : list[int]
        List of window sizes for rolling calculations.
    lags : list[int]
        List of lag values for lagged features.

    Returns
    -------
    pd.DataFrame
        DataFrame with games data enhanced with team statistics features:
        - Rolling averages and std: rolling_avg_{stat}_{window}days, rolling_std_{stat}_{window}days
        - Lagged features: {stat}_lag{lag}
        - Home team features: home_{feature}
        - Away team features: away_{feature}  
        - Home-away differences: {feature}_diff
        - Home/away ratios: {feature}_ratio
        
    Features generated for these statistics:
        - obp (on-base percentage)
        - slg (slugging percentage) 
        - era (earned run average)
        - whip (walks + hits per innings pitched)
        - strikeouts_pitching
        - baseonballs_pitching  
        - hits_pitching

    Note
        Assumes df_team_stats is already sorted by team_id and game_date_time.
        Home/away differences and ratios are only calculated for rolling features, not lags.
    """
    # Team statistics to consider for feature engineering
    team_stat_cols = ['obp','slg','era','whip','strikeouts_pitching','baseonballs_pitching','hits_pitching']
    
    # Create rolling features
    df_team_stats_rolling = get_rolling_team_stat_features(
        df_team_stats, 
        team_stat_cols=team_stat_cols, 
        window=window
    )
    rolling_avg_team_stat_cols = [f"rolling_avg_{stat}_{w}days" for stat in team_stat_cols for w in window]
    rolling_std_team_stat_cols = [f"rolling_std_{stat}_{w}days" for stat in team_stat_cols for w in window]
    rolling_team_stat_cols = rolling_avg_team_stat_cols + rolling_std_team_stat_cols

    # Create lagged features
    df_team_stats_rolling_lags = get_lag_team_stat_features(
        df_team_stats_rolling,
        team_stat_cols=team_stat_cols,
        lags=lags
    )
    lags_team_stat_cols = [f"{stat}_lag{lag}" for stat in team_stat_cols for lag in lags]
    rolling_lags_team_stat_cols = rolling_team_stat_cols + lags_team_stat_cols
    
    # Merge new rolling + lagged features into games data
    df_games = merge_team_stats_features_into_games(
        df_games=df_games, 
        df_team_stats=df_team_stats_rolling_lags,
        team_stats_cols=rolling_lags_team_stat_cols
    )

    # Create home - away (difference) and home / away (ratio) features
    for col in rolling_team_stat_cols:
        df_games[f"{col}_diff"] = df_games[f"home_{col}"] - df_games[f"away_{col}"]
        df_games[f"{col}_ratio"] = df_games[f"home_{col}"] / (df_games[f"away_{col}"] + 1e-8) 

    return df_games