from typing import Literal
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
    home_schedule = (
        df[['home_team',date_col, date_time_col]]
        .rename(columns={'home_team': 'team'})
        .assign(home_ind=1)
    )
    away_schedule = (
        df[['away_team', date_col, date_time_col]]
        .rename(columns={'away_team': 'team'})
        .assign(home_ind=0)
    )

    # Join them into one 'team' column
    team_schedule = (
        pd.concat([home_schedule, away_schedule])
        .sort_values(['team',date_time_col])
    )

    return team_schedule

def team_rest_days(
        df: pd.DataFrame,
        date_time_col: str = 'game_date_time',
        date_col: str = 'game_date'
        ) -> pd.Series:
    """
    DOES NOT WORK AS INTENDED -- NEED TO COMBINE HOME AND AWAY SCHEDULES FOR EACH TEAM!!
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
    date_col : str, default 'game_date'
        Column name containing date information used to calculate day differences.
        
    Returns
    -------
    pd.Series
        Series with the same index as input DataFrame containing the number of
        rest days for each game. First game for each team will have 0 rest days.
    """
    #team_col = f"{team}_team"
    df_team_sched = team_schedule(df)
    return (
        df_team_sched
        #.sort_values(['team',date_time_col])
        .groupby('team')[date_col]
        .diff()
        .dt.days
        .fillna(value=0)
        -1
    ).clip(lower=0)

def team_rest_7day_avg(
        df: pd.DataFrame, 
        team: Literal['home','away'],
        date_col: str = 'game_date'
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
    team_col = f"{team}_team"
    return (
        df
        .sort_values('game_date')
        .groupby(team_col)[f'{team}_rest_days']
        .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    )
    
def create_rest_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df['home_rest_days'] = team_rest_days(df, team='home')
    df['away_rest_days'] = team_rest_days(df, team='away')

    df['home_back2back'] = (df['home_rest_days'] <= 1).astype(int)
    df['away_back2back'] = (df['away_rest_days'] <= 1).astype(int)

    df['home_rest_7day_avg'] = team_rest_7day_avg(df, team='home')
    df['away_rest_7day_avg'] = team_rest_7day_avg(df, team='away')

    df['rest_difference'] = df['home_rest_days'] - df['away_rest_days']

    return df


def create_date_time_features(df: pd.DataFrame) -> pd.DataFrame:
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

def team_rolling_avg(
        df: pd.DataFrame, 
        team: Literal['home','away'],
        value_col: str,
        window: int = 3,
        date_time_col: str = 'game_date_time'
        ) -> pd.Series:
    team_col = f"{team}_team"
    return (
        df
        .sort_values(date_time_col)
        .groupby(team_col)[value_col]
        .transform(lambda x: x.shift(1).rolling(window, min_period=1).mean())
    )



def create_team_rolling_avg(df: pd.DataFrame) -> pd.DataFrame:

    ## Home/Away Team Splits
    # Runs scored
    df['home_score_3day_avg'] = team_rolling_avg(df, team='home', value_col='home_score',window=3)
    df['away_score_3day_avg'] = team_rolling_avg(df, team='away', value_col='away_score',window=3)
    df['home_score_7day_avg'] = team_rolling_avg(df, team='home', value_col='home_score',window=7)
    df['away_score_7day_avg'] = team_rolling_avg(df, team='away', value_col='away_score',window=7)

    # Runs allowed
    df['home_runs_allowed_3day_avg'] = team_rolling_avg(df, team='home', value_col='away_score')
    # Run differential



    return df

    