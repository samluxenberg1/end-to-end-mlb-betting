import pandas as pd

def team_schedule(
    df: pd.DataFrame, 
    date_col: str = 'game_date', 
    date_time_col: str = 'game_date_time'
) -> pd.DataFrame:
     
    """
    Transform a games DataFrame into a team-centric schedule view.
    
    Takes a DataFrame with game data (home_team vs away_team format) and converts it
    into a schedule where each row represents one team's game, with an indicator
    for whether they played at home or away.
    
    Args:
        df (pd.DataFrame): DataFrame containing game data with columns:
            - 'home_team': Name of the home team
            - 'away_team': Name of the away team
            - date_col: Column name containing game dates
            - date_time_col: Column name containing game datetime stamps
        date_col (str, optional): Name of the date column. Defaults to 'game_date'.
        date_time_col (str, optional): Name of the datetime column. Defaults to 'game_date_time'.
    
    Returns:
        pd.DataFrame: Team schedule DataFrame with columns:
            - 'team': Team name
            - date_col: Game date (same as input)
            - date_time_col: Game datetime (same as input)
            - 'home_ind': Binary indicator (1 if home game, 0 if away game)
            
        The DataFrame is sorted by team name and then by game datetime.
     """
    # Extract home and away games for every team
    common_cols = [date_col, date_time_col, 'home_score','away_score']
    home_cols = ['home_team'] + common_cols
    home_schedule = (
        df[home_cols]
        .rename(columns={
            'home_team': 'team',
            'home_score': 'team_score',
            'away_score': 'opp_score'
        })
        .assign(home_ind=1)
    )
    away_cols = ['away_team'] + common_cols
    away_schedule = (
        df[away_cols]
        .rename(columns={
            'away_team': 'team',
            'home_score': 'opp_score',
            'away_score': 'team_score'
        })
        .assign(home_ind=0)
    )

    # Join them into one 'team' column
    team_schedule = (
        pd.concat([home_schedule, away_schedule])
        .sort_values(['team',date_time_col])
    )

    # Create team win column
    team_schedule['team_win'] = (team_schedule['team_score'] > team_schedule['opp_score']).astype(int)

    # Create run differential column (team_score - opp_score)
    team_schedule['team_run_diff'] = team_schedule['team_score'] - team_schedule['opp_score']

    return team_schedule

def team_rest_days(
        df: pd.DataFrame,
        date_time_col: str = 'game_date_time',
        date_col: str = 'game_date'
        ) -> pd.Series:
    """
    Calculate the number of rest days between consecutive games for each team.
    
    This function sorts games chronologically and calculates how many days of rest
    each team had before each game by finding the difference between consecutive
    game dates for each team.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing game data with team and date information.
    team : {'home', 'away'}
        Specifies whether to calculate rest days for home teams or away teams.
        Will look for corresponding '{team}_team' column in the DataFrame.
    date_time_col : str, default 'game_date_time'
        Column name containing datetime information used for chronological sorting. 
        Necessary for sorting between double headers for a single team.
    date_col : str, default 'game_date'
        Column name containing date information used to calculate day differences.
        
    Returns
    -------
    pd.Series
        Series with the same index as input DataFrame containing the number of
        rest days for each game. First game for each team will have 0 rest days.
    """
    df_team_sched = team_schedule(df, date_time_col=date_time_col, date_col=date_col)
    return (
        df_team_sched
        .groupby('team')[date_col]
        .diff()
        .dt.days
        .fillna(value=0)
        -1
    ).clip(lower=0)

def team_games_previous_7days(
        df: pd.DataFrame, 
        date_col: str = 'game_date',
        date_time_col: str = 'game_date_time'
        ) -> pd.Series:
    """
    Calculate the 7-day rolling average of rest days for each team.
    
    This function computes a rolling average of rest days over the previous 7 games
    for each team, providing a smoothed measure of how well-rested teams have been
    over recent games. Uses transform to maintain the original DataFrame index.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing game data with team information and rest days.
        Must include a column named '{team}_rest_days' (e.g., 'home_rest_days').
    team : {'home', 'away'}
        Specifies whether to calculate rolling average for home or away teams.
        Will look for '{team}_team' and '{team}_rest_days' columns.
    date_col : str, default 'game_date'
        Column name containing date information used for chronological sorting.
        
    Returns
    -------
    pd.Series
        Series with the same index as input DataFrame containing the 7-day
        rolling average of rest days for each team and game.
    """
    df_team_sched = team_schedule(df, date_time_col=date_time_col, date_col=date_col)
    
    return (
        df_team_sched
        .groupby('team')
        .rolling(window='7D', on=date_time_col, min_periods=1, closed='left')
        .count()['game_date']
        .fillna(value=0)
        .reset_index(drop=True,level=0)
    )
def merge_team_features_into_games(df_games: pd.DataFrame, df_team_schedule: pd.DataFrame, team_schedule_cols: list[str], date_time_col: str = 'game_date_time'):
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

def get_schedule_features(df: pd.DataFrame, date_col: str = 'game_date', date_time_col: str = 'game_date_time'):
    # create team schedule data
    df_team_sched = team_schedule(df, date_col=date_col, date_time_col=date_time_col)
    # add team_rest_days
    df_team_sched['team_rest_days'] = team_rest_days(df)
    # add rest_days_7day_avg
    df_team_sched['team_games_prev_7days'] = team_games_previous_7days(df)

    # Merge
    team_schedule_cols = ['team','team_rest_days','team_games_prev_7days'] # no date_time_col here -- add separately
    df_games = merge_team_features_into_games(
        df_games=df, 
        df_team_schedule=df_team_sched,
        team_schedule_cols=team_schedule_cols, 
        date_time_col=date_time_col
        )

    return df_games

def team_win_rate_last_10(
    df: pd.DataFrame,
    date_col: str = 'game_date',
    date_time_col: str = 'game_date_time'
):
    df_team_sched = team_schedule(df,date_time_col=date_time_col, date_col=date_col)
    return (
        df_team_sched
        .groupby('team')['team_win']
        .rolling(window=10, min_periods=1, closed='left')
        .mean()
        .fillna(value=0)
        .reset_index(drop=True, level=0)
    )

def team_avg_run_diff_last_10(
        df: pd.DataFrame,
        date_col: str = 'game_date',
        date_time_col: str = 'game_date_time'
):
    df_team_sched = team_schedule(df, date_time_col=date_time_col, date_col=date_col)
    return (
        df_team_sched
        .groupby('team')['team_run_diff']
        .rolling(window=10, min_periods=1, closed='left')
        .mean()
        .fillna(value=0)
        .reset_index(drop=True, level=0)
    )
def team_std_run_diff_last_10(
        df: pd.DataFrame,
        date_col: str = 'game_date',
        date_time_col: str = 'game_date_time'
):
    df_team_sched = team_schedule(df, date_time_col=date_time_col, date_col=date_col)
    return (
        df_team_sched
        .groupby('team')['team_run_diff']
        .rolling(window=10, min_periods=1, closed='left')
        .std()
        .fillna(value=0)
        .reset_index(drop=True, level=0)
    )
def team_avg_runs_score_last_10(
        df: pd.DataFrame,
        date_col: str = 'game_date',
        date_time_col: str = 'game_date_time'
):
    df_team_sched = team_schedule(df, date_time_col=date_time_col, date_col=date_col)
    return (
        df_team_sched
        .groupby('team')['team_score']
        .rolling(window=10, min_periods=1, closed='left')
        .mean()
        .fillna(value=0)
        .reset_index(drop=True, level=0)
    ) 
def team_std_runs_score_last_10(
        df: pd.DataFrame,
        date_col: str = 'game_date',
        date_time_col: str = 'game_date_time'
):
    df_team_sched = team_schedule(df, date_time_col=date_time_col, date_col=date_col)
    return (
        df_team_sched
        .groupby('team')['team_score']
        .rolling(window=10, min_periods=1, closed='left')
        .std()
        .fillna(value=0)
        .reset_index(drop=True, level=0)
    ) 
def team_avg_runs_allowed_last_10(
        df: pd.DataFrame,
        date_col: str = 'game_date',
        date_time_col: str = 'game_date_time'
):
    df_team_sched = team_schedule(df, date_time_col=date_time_col, date_col=date_col)
    return (
        df_team_sched
        .groupby('team')['opp_score']
        .rolling(window=10, min_periods=1, closed='left')
        .mean()
        .fillna(value=0)
        .reset_index(drop=True, level=0)
    ) 
def team_std_runs_allowed_last_10(
        df: pd.DataFrame,
        date_col: str = 'game_date',
        date_time_col: str = 'game_date_time'
):
    df_team_sched = team_schedule(df, date_time_col=date_time_col, date_col=date_col)
    return (
        df_team_sched
        .groupby('team')['opp_score']
        .rolling(window=10, min_periods=1, closed='left')
        .std()
        .fillna(value=0)
        .reset_index(drop=True, level=0)
    ) 

def get_outcome_features(
        df: pd.DataFrame,
        date_col: str = 'game_date',
        date_time_col: str = 'game_date_time'
):
    # Create team schedule
    df_team_sched = team_schedule(df, date_time_col=date_time_col, date_col=date_col)

    # Win rate last 10 games
    df_team_sched['win_rate_last_10'] = team_win_rate_last_10(df, date_time_col=date_time_col, date_col=date_col)

    # Run differential last 10 games
    df_team_sched['avg_run_diff_last_10'] = team_avg_run_diff_last_10(df, date_time_col=date_time_col, date_col=date_col)
    df_team_sched['std_run_diff_last_10'] = team_std_run_diff_last_10(df, date_time_col=date_time_col, date_col=date_col)

    # Runs scored last 10 games
    df_team_sched['avg_runs_scored_last_10'] = team_avg_runs_score_last_10(df, date_time_col=date_time_col, date_col=date_col)
    df_team_sched['std_runs_scored_last_10'] = team_std_runs_score_last_10(df, date_time_col=date_time_col, date_col=date_col)

    # Runs allowed last 10 games
    df_team_sched['avg_runs_allowed_last_10'] = team_avg_runs_allowed_last_10(df, date_time_col=date_time_col, date_col=date_col)
    df_team_sched['std_runs_allowed_last_10'] = team_std_runs_allowed_last_10(df, date_time_col=date_time_col, date_col=date_col)

    # Merge
    team_schedule_cols = [
        'team', 
        'win_rate_last_10',
        'avg_run_diff_last_10','std_run_diff_last_10',
        'avg_runs_scored_last_10','std_runs_scored_last_10',
        'avg_runs_allowed_last_10','std_runs_allowed_last_10'
        ]
    df_games = merge_team_features_into_games(
        df_games=df, 
        df_team_schedule=df_team_sched,
        team_schedule_cols=team_schedule_cols,
        date_time_col=date_time_col
        )

    return df_games

def get_date_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create date and time-based features from game datetime information.
    
    Extracts temporal features that can be useful for analysis or modeling,
    including seasonal patterns (month), weekly patterns (day of week), and
    daily patterns (hour of game).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing game data. Must include 'game_date' and 
        'game_date_time' columns with datetime data types.
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with the following new columns added:
        - 'game_month': Month of the game (1-12, where 1=January)
        - 'game_day_of_week': Day of week (0-6, where 0=Monday, 6=Sunday)
        - 'game_hour': Hour of game start time (0-23 in 24-hour format)
    """
    df['game_month'] = df['game_date'].dt.month
    df['game_day_of_week'] = df['game_date'].dt.dayofweek
    df['game_hour'] = df['game_date_time'].dt.hour

    return df  

def create_game_features(df: pd.DataFrame, date_col: str = 'game_date',date_time_col: str = 'game_date_time') -> pd.DataFrame:
    """
    Create comprehensive rest-related features for both home and away teams.
    
    This function generates multiple rest-based features that can be used for
    analysis or modeling, including rest days, back-to-back indicators,
    rolling averages, and relative rest advantages between teams.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing game data. Must include 'home_team', 'away_team',
        'game_date_time', and 'game_date' columns.
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with the following new columns added:
        - 'home_rest_days': Days of rest for home team before each game
        - 'away_rest_days': Days of rest for away team before each game  
        - 'home_back2back': Binary indicator (1 if home team has ≤1 rest day)
        - 'away_back2back': Binary indicator (1 if away team has ≤1 rest day)
        - 'home_rest_7day_avg': 7-game rolling average of home team rest days
        - 'away_rest_7day_avg': 7-game rolling average of away team rest days
        - 'rest_difference': Home rest days minus away rest days
    """
    # Schedule Features
    df_games = get_schedule_features(df, date_col=date_col, date_time_col=date_time_col)

    df_games['home_back2back'] = (df_games['home_team_rest_days'] <= 1).astype(int)
    df_games['away_back2back'] = (df_games['away_team_rest_days'] <= 1).astype(int)

    df_games['rest_difference'] = df_games['home_team_rest_days'] - df_games['away_team_rest_days']

    # Outcome Features
    df_games = get_outcome_features(df_games, date_col=date_col, date_time_col=date_time_col)

    # Date-Time Features
    df_games = get_date_time_features(df_games)

    return df_games


