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
        date_col (str, optional): Name of the date column. Defaults to 'game_date'.
        date_time_col (str, optional): Name of the datetime column. Defaults to 'game_date_time'.
    
    Returns:
        DataFrame containing scoring, home indicator, and win columns.
            
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
        .reset_index(drop=True)
    )

    # Create team win column
    team_schedule['team_win'] = (team_schedule['team_score'] > team_schedule['opp_score']).astype(int)

    # Create run differential column (team_score - opp_score)
    team_schedule['team_run_diff'] = team_schedule['team_score'] - team_schedule['opp_score']

    # Create season for grouping
    team_schedule['season'] = team_schedule['game_date'].dt.year

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
        .groupby(['team','seaon'])[date_col]
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
    Calculate the 7-day rolling sum of rest days for each team prior to the current date.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing game data with team information and rest days.
        Must include a column named '{team}_rest_days' (e.g., 'home_rest_days').
    date_col : str, default 'game_date'
        Column name containing date information used for chronological sorting.
    date_time_col : str, default 'game_date_time'
        Column name containing datetime information used for chronological sorting. 
        Necessary for sorting between double headers for a single team.
        
    Returns
    -------
    pd.Series
        Series with the same index as input DataFrame containing the 7-day
        rolling sum of rest days for each team and game.
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
        Games DataFrame with team features merged in, where each feature from team_schedule_cols appears twice - once prefixed with 'home_' and once prefixed with 'away_' for respective teams.
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

def get_schedule_features(
        df: pd.DataFrame, 
        date_col: str = 'game_date', 
        date_time_col: str = 'game_date_time'
        ) -> pd.DataFrame:
    """
    Generate schedule-based features for games DataFrame.

    Creates team-level schedule features including rest days and recent game frequency, 
    then merges these features back into the original games DataFrame for both home
    and away teams.

    Parameters
    ----------
    df: pd.DataFrame
        Input games DataFrame.
    date_col: str, default 'game_date'
        Column name containing date information.
    date_time_col: str, default 'game_date_time'
        Column name containing datetime information.

    Returns 
    -------
    pd.DataFrame
        Games DataFrame with schedule features added: 
        - 'home_team_rest_days': Rest days for home team before current game
        - 'away_team_rest_days': Rest days for away team before current game
        - 'home_team_games_prev_7days': Number of games home team played in previous 7 days
        - 'away_team_games_prev_7days': Number of games away team played in previous 7 days
    """
    # Create team schedule data
    df_team_sched = team_schedule(df, date_col=date_col, date_time_col=date_time_col)
    # Add team_rest_days
    df_team_sched['team_rest_days'] = team_rest_days(df)
    # Add rest_days_7day_avg
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
) -> pd.Series:
    """
    Calculate each team's win rate over their last 10 games.

    Computes a rolling win rate using a 10-game window for each team, providing a measure of recent team performance. 
    Uses closed='left' to avoid look-ahead bias.

    Parameters
    ----------
    df : pd.DataFrame
        Input games DataFrame.
    date_col: str, default 'game_date'
        Column name containing date information.
    date_time_col: str, default 'game_date_time'
        Column name containing datetime information.

    Returns
    -------
    pd.Series
        Series containing win rate over last 10 games for each team and date.
        Values range from 0.0 (no wins) to 1.0 (all wins). For teams with fewer than 10 games, uses all available games. 
        Values start over at 0.0 at the beginning of each season.
    """
    df_team_sched = team_schedule(df,date_time_col=date_time_col, date_col=date_col)
    return (
        df_team_sched
        .groupby(['team','season'])['team_win']
        .rolling(window=10, min_periods=1, closed='left')
        .mean()
        .fillna(value=0)
        .reset_index(drop=True)
    )

def team_avg_run_diff_last_10(
        df: pd.DataFrame,
        date_col: str = 'game_date',
        date_time_col: str = 'game_date_time'
) -> pd.Series:
    """
    Calculate each team's average run differential over their last 10 games.

    Computes a rolling average of run differential (runs scored - runs allowed)
    using a 10-game window for each team.

    Parameters
    ----------
    df : pd.DataFrame
        Input games DataFrame.
    date_col : str, default 'game_date'
        Column name containing date information.
    date_time_col : str, default 'game_date_time'
        Column name containing datetime information.

    Returns 
    -------
    pd.Series
        Series containing average run differential over last 10 games for each team and date. 
        For teams with fewer than 10 games, uses all available games. 
        Values start over at 0.0 at the beginning of each season.
    """
    df_team_sched = team_schedule(df, date_time_col=date_time_col, date_col=date_col)
    return (
        df_team_sched
        .groupby(['team','season'])['team_run_diff']
        .rolling(window=10, min_periods=1, closed='left')
        .mean()
        .fillna(value=0)
        .reset_index(drop=True)
    )
def team_std_run_diff_last_10(
        df: pd.DataFrame,
        date_col: str = 'game_date',
        date_time_col: str = 'game_date_time'
) -> pd.Series:
    """
    Calculate each team's standard deviation run differential over their last 10 games.

    Computes a rolling standard deviation of run differential (runs scored - runs allowed)
    using a 10-game window for each team.

    Parameters
    ----------
    df : pd.DataFrame
        Input games DataFrame.
    date_col : str, default 'game_date'
        Column name containing date information.
    date_time_col : str, default 'game_date_time'
        Column name containing datetime information.

    Returns 
    -------
    pd.Series
        Series containing standard deviation run differential over last 10 games for each team and date. 
        For teams with fewer than 10 games, uses all available games. 
        Values start over at 0.0 at the beginning of each season.
    """
    df_team_sched = team_schedule(df, date_time_col=date_time_col, date_col=date_col)
    return (
        df_team_sched
        .groupby(['team','season'])['team_run_diff']
        .rolling(window=10, min_periods=1, closed='left')
        .std()
        .fillna(value=0)
        .reset_index(drop=True)
    )
def team_avg_runs_score_last_10(
        df: pd.DataFrame,
        date_col: str = 'game_date',
        date_time_col: str = 'game_date_time'
) -> pd.Series:
    """
    Calculate each team's average runs scored over their last 10 games.

    Computes a rolling average of runs scored 
    using a 10-game window for each team.

    Parameters
    ----------
    df : pd.DataFrame
        Input games DataFrame.
    date_col : str, default 'game_date'
        Column name containing date information.
    date_time_col : str, default 'game_date_time'
        Column name containing datetime information.

    Returns 
    -------
    pd.Series
        Series containing average runs scored over last 10 games for each team and date. 
        For teams with fewer than 10 games, uses all available games. 
        Values start over at 0.0 at the beginning of each season.
    """
    df_team_sched = team_schedule(df, date_time_col=date_time_col, date_col=date_col)
    return (
        df_team_sched
        .groupby(['team','season'])['team_score']
        .rolling(window=10, min_periods=1, closed='left')
        .mean()
        .fillna(value=0)
        .reset_index(drop=True)
    ) 
def team_std_runs_score_last_10(
        df: pd.DataFrame,
        date_col: str = 'game_date',
        date_time_col: str = 'game_date_time'
) -> pd.Series:
    """
    Calculate each team's standard deviation runs scored over their last 10 games.

    Computes a rolling standard deviation of runs scored 
    using a 10-game window for each team.

    Parameters
    ----------
    df : pd.DataFrame
        Input games DataFrame.
    date_col : str, default 'game_date'
        Column name containing date information.
    date_time_col : str, default 'game_date_time'
        Column name containing datetime information.

    Returns 
    -------
    pd.Series
        Series containing standard deviation runs scored over last 10 games for each team and date. 
        For teams with fewer than 10 games, uses all available games. 
        Values start over at 0.0 at the beginning of each season.
    """
    df_team_sched = team_schedule(df, date_time_col=date_time_col, date_col=date_col)
    return (
        df_team_sched
        .groupby(['team','season'])['team_score']
        .rolling(window=10, min_periods=1, closed='left')
        .std()
        .fillna(value=0)
        .reset_index(drop=True)
    ) 
def team_avg_runs_allowed_last_10(
        df: pd.DataFrame,
        date_col: str = 'game_date',
        date_time_col: str = 'game_date_time'
) -> pd.Series:
    """
    Calculate each team's average runs allowed over their last 10 games.

    Computes a rolling average of runs allowed 
    using a 10-game window for each team.

    Parameters
    ----------
    df : pd.DataFrame
        Input games DataFrame.
    date_col : str, default 'game_date'
        Column name containing date information.
    date_time_col : str, default 'game_date_time'
        Column name containing datetime information.

    Returns 
    -------
    pd.Series
        Series containing average runs allowed over last 10 games for each team and date. 
        For teams with fewer than 10 games, uses all available games. 
        Values start over at 0.0 at the beginning of each season.
    """
    df_team_sched = team_schedule(df, date_time_col=date_time_col, date_col=date_col)
    return (
        df_team_sched
        .groupby(['team','season'])['opp_score']
        .rolling(window=10, min_periods=1, closed='left')
        .mean()
        .fillna(value=0)
        .reset_index(drop=True)
    ) 
def team_std_runs_allowed_last_10(
        df: pd.DataFrame,
        date_col: str = 'game_date',
        date_time_col: str = 'game_date_time'
) -> pd.Series:
    """
    Calculate each team's standard deviation runs allowed over their last 10 games.

    Computes a rolling standard deviation of runs allowed 
    using a 10-game window for each team.

    Parameters
    ----------
    df : pd.DataFrame
        Input games DataFrame.
    date_col : str, default 'game_date'
        Column name containing date information.
    date_time_col : str, default 'game_date_time'
        Column name containing datetime information.

    Returns 
    -------
    pd.Series
        Series containing standard deviation runs allowed over last 10 games for each team and date. 
        For teams with fewer than 10 games, uses all available games. 
        Values start over at 0.0 at the beginning of each season.
    """
    df_team_sched = team_schedule(df, date_time_col=date_time_col, date_col=date_col)
    return (
        df_team_sched
        .groupby(['team','season'])['opp_score']
        .rolling(window=10, min_periods=1, closed='left')
        .std()
        .fillna(value=0)
        .reset_index(drop=True)
    ) 

def get_outcome_features(
        df: pd.DataFrame,
        date_col: str = 'game_date',
        date_time_col: str = 'game_date_time'
) -> pd.DataFrame:
    """
    Generate outcome features based on recent game outcomes.

    Creates multiple rolling statistics over the last 10 games for each team, including
    win rates, run differentials, runs scored, and runs allowed. These features capture
    both central tendencies (means) and variability (standard deviation) of team 
    performance.

    Parameters
    ----------
    df : pd.DataFrame
        Input games DataFrame.
    date_col : str, default 'game_date'
        Column name containing date information.
    date_time_col : str, default 'game_date_time'
        Column name containing datetime information.

    Returns 
    -------
    pd.DataFrame
        Games DataFrame with team outcome features added for both home and away teams: 
        - Win rate over last 10 games
        - Average and standard deviation of run differential over last 10 games
        - Average and standard deviation of runs scored over last 10 games
        - Average and standard deviation of runs allowed over last 10 games
    """
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
    Create comprehensive game features including schedule, outcome, and temporate features.

    This is the main feature engineering function that generates multiple categories of 
    features. It combines schedule-based features (rest days, game frequency), 
    outcome-based features (recent performance metrics), and temporal features 
    (seasonal and time-of-day patterns).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input games DataFrame.
    date_col : str, default 'game_date'
        Column name containing date information.
    date_time_col : str, default 'game_date_time'
        Column name containing datetime information.
        
    Returns
    -------
    pd.DataFrame
        Original games DataFrame with the following new columns added:
        
        Schedule Features:
        - 'home_team_rest_days': Days of rest for home team before each game
        - 'away_team_rest_days': Days of rest for away team before each game
        - 'home_team_games_prev_7days': Games played by home team in previous 7 days
        - 'away_team_games_prev_7days': Games played by away team in previous 7 days
        - 'home_back2back': Binary indicator (1 if home team has ≤1 rest day)
        - 'away_back2back': Binary indicator (1 if away team has ≤1 rest day)
        - 'rest_difference': Home rest days minus away rest days
        
        Outcome Features (last 10 games for each team):
        - 'home_win_rate_last_10': Home team win rate over last 10 games
        - 'away_win_rate_last_10': Away team win rate over last 10 games
        - 'home_avg_run_diff_last_10': Home team average run differential
        - 'away_avg_run_diff_last_10': Away team average run differential
        - 'home_std_run_diff_last_10': Home team run differential standard deviation
        - 'away_std_run_diff_last_10': Away team run differential standard deviation
        - 'home_avg_runs_scored_last_10': Home team average runs scored
        - 'away_avg_runs_scored_last_10': Away team average runs scored
        - 'home_std_runs_scored_last_10': Home team runs scored standard deviation
        - 'away_std_runs_scored_last_10': Away team runs scored standard deviation
        - 'home_avg_runs_allowed_last_10': Home team average runs allowed
        - 'away_avg_runs_allowed_last_10': Away team average runs allowed
        - 'home_std_runs_allowed_last_10': Home team runs allowed standard deviation
        - 'away_std_runs_allowed_last_10': Away team runs allowed standard deviation
        
        Temporal Features:
        - 'game_month': Month of the game (1-12, where 1=January)
        - 'game_day_of_week': Day of week (0-6, where 0=Monday, 6=Sunday)
        - 'game_hour': Hour of game start time (0-23 in 24-hour format)
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


