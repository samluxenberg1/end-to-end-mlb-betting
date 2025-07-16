import pandas as pd

def merge_team_features_into_games(
        df_games: pd.DataFrame, 
        df_team_schedule: pd.DataFrame, 
        team_schedule_cols: list[str], 
        date_time_col: str = 'game_date_time'
        ) -> pd.DataFrame:
    """
    Merge team-level features into the games DataFrame for both home and away teams.

    Parameters
    ----------
    df_games : pd.DataFrame
        Original games DataFrame.
    df_team_schedule : pd.DataFrame
        Team schedule DataFrame containing team-level features.
    team_schedule_cols : list[str]
        List of column names from df_team_schedule to merge into df_games.
    date_time_col : str, default 'game_date_time'
        Column name containing datetime infromation used for merging. 
    
    Returns
    -------
    pd.DataFrame
        Games DataFrame with team features merged in, where each feature 
        from team_schedule_cols appears twice - once prefixed with 'home_' 
        and once prefixed with 'away_' for respective teams.
    """
    # Home Merge
    df_games = df_games.merge(
        df_team_schedule[team_schedule_cols + [date_time_col]],
        how='left',
        left_on=['home_team', date_time_col],
        right_on=['team', date_time_col]
    )
    df_games.drop('team', axis=1, inplace=True)
    home_cols_rename = {col: f'home_{col}' for col in team_schedule_cols}
    df_games.rename(columns=home_cols_rename, inplace=True)

    # Away Merge
    df_games = df_games.merge(
        df_team_schedule[team_schedule_cols + [date_time_col]],
        how='left',
        left_on=['away_team',date_time_col],
        right_on=['team',date_time_col]
    )
    df_games.drop('team', axis=1, inplace=True)
    away_cols_rename = {col: f'away_{col}' for col in team_schedule_cols}
    df_games.rename(columns=away_cols_rename, inplace=True)

    return df_games

def merge_team_stats_features_into_games(
        df_games: pd.DataFrame,
        df_team_stats: pd.DataFrame,
        team_stats_cols: list[str],
        date_time_col: str = 'game_date_time'
) -> pd.DataFrame:
    """Note: This is almost the same as the previous merge function. Refactor later when there's time...."""
    # Home Merge
    df_games = df_games.merge(
        df_team_stats[team_stats_cols + ['team_id',date_time_col]],
        how='left',
        left_on=['home_team_id', date_time_col],
        right_on=['team_id', date_time_col]
    )
    df_games.drop('team_id', axis=1, inplace=True)
    home_cols_rename = {col: f'home_{col}' for col in team_stats_cols}
    df_games.rename(columns=home_cols_rename, inplace=True)

    # Away Merge
    df_games = df_games.merge(
        df_team_stats[team_stats_cols + ['team_id',date_time_col]],
        how='left',
        left_on=['away_team_id', date_time_col],
        right_on=['team_id', date_time_col]
    )
    df_games.drop('team_id', axis=1, inplace=True)
    away_cols_rename = {col: f'away_{col}' for col in team_stats_cols}
    df_games.rename(columns=away_cols_rename, inplace=True)

    return df_games

def merge_games_identifiers_into_team_stats(
        df_games: pd.DataFrame, 
        df_team_stats: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge Strategy + Cleaning

    1. Split home and away
    2. Merge games identifiers separately
    3. Concatenate and sort
    4. Drop rows with no matching game_id from df_games
    5. Add 'season' as column
    """
    # 1. Split home and away
    df_team_stats_home = df_team_stats.query("team_side=='home'").copy()
    df_team_stats_away = df_team_stats.query("team_side=='away'").copy()

    # 2. Merge games identifiers separately
    common_game_cols = ['game_id','game_date','game_date_time']
    home_game_cols = common_game_cols + ['home_team_id']
    away_game_cols = common_game_cols + ['away_team_id']
    # Home Merge
    df_team_stats_home = df_team_stats_home.merge(
        df_games[home_game_cols],
        how='left',
        left_on=['game_pk','team_id'],
        right_on=['game_id','home_team_id']
    )
    df_team_stats_home.drop('home_team_id', axis=1, inplace=True)
    # Away Merge
    df_team_stats_away = df_team_stats_away.merge(
        df_games[away_game_cols],
        how='left',
        left_on=['game_pk','team_id'],
        right_on=['game_id','away_team_id']
    )
    df_team_stats_away.drop('away_team_id', axis=1, inplace=True)

    # 3. Concatenate and sort
    df_team_stats = (
        pd.concat([df_team_stats_home, df_team_stats_away], ignore_index=True)
        .sort_values(['team_id','game_date_time'])
        .reset_index(drop=True)
    )

    # 4. Drop rows with not matching game_id from games
    df_team_stats = df_team_stats[~df_team_stats['game_id'].isna()]

    # 5. Add season
    df_team_stats['season'] = df_team_stats['game_date'].dt.year.astype('int64')    

    return df_team_stats