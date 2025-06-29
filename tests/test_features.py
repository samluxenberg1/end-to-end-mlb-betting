import pytest
import pandas as pd
from features.games_features import (
    team_schedule,
    team_rest_days,
    team_games_previous_7days, 
    team_win_rate_last_10,
    merge_team_features_into_games,
    get_schedule_features
)

def test_team_schedule():
    """Test that team_schedule correctly transforms games into team schedules."""
    # Input data
    games_df = pd.DataFrame({
        'home_team': ['New York Yankees', 'Boston Red Sox'],
        'away_team': ['Boston Red Sox', 'Los Angeles Dodgers'],
        'game_date': pd.to_datetime(['2024-07-15', '2024-07-16']),
        'game_date_time': pd.to_datetime(['2024-07-15 19:05:00', '2024-07-16 19:10:00']),
        'home_score': [5, 8],
        'away_score': [3, 6]
    })
    
    # Expected output (sorted by team, then date_time)
    expected = pd.DataFrame({
        'team': ['Boston Red Sox', 'Boston Red Sox', 'Los Angeles Dodgers', 'New York Yankees'],
        'game_date': pd.to_datetime(['2024-07-15', '2024-07-16', '2024-07-16', '2024-07-15']),
        'game_date_time': pd.to_datetime(['2024-07-15 19:05:00', '2024-07-16 19:10:00', 
                                        '2024-07-16 19:10:00', '2024-07-15 19:05:00']),
        'team_score': [3, 8, 6, 5],
        'opp_score': [5, 6, 8, 3],
        'home_ind': [0, 1, 0, 1],
        'team_win': [0, 1, 0, 1],
        'team_run_diff': [-2, 2, -2, 2],
        'season': [2024, 2024, 2024, 2024]
    })
    
    result = team_schedule(games_df)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

def test_team_rest_days():
    """Test that team_rest_days correctly calculates rest days between games."""
    # Input data - same team playing consecutive games
    games_df = pd.DataFrame({
        'home_team': ['Team A', 'Team B', 'Team A'],
        'away_team': ['Team B', 'Team A', 'Team C'], 
        'game_date': pd.to_datetime(['2024-07-15', '2024-07-17', '2024-07-18']),
        'game_date_time': pd.to_datetime(['2024-07-15 19:00:00', '2024-07-17 19:00:00', '2024-07-18 19:00:00']),
        'home_score': [5, 3, 7],
        'away_score': [2, 4, 1]
    })
    
    # Expected: Team A plays 7/15, 7/17, 7/18 -> rest days: 0, 1, 0
    #          Team B plays 7/15, 7/17 -> rest days: 0, 1
    #          Team C plays 7/18 -> rest days: 0
    expected = pd.Series([0, 1, 0, 0, 1, 0], dtype='float64')
    
    result = team_rest_days(games_df)
    print(result)
    pd.testing.assert_series_equal(
        result.reset_index(drop=True), 
        expected.reset_index(drop=True), 
        check_names=False
        )
    
def test_team_games_previous_7days():
    """Test that team_games_previous_7days correctly counts games in rolling 7-day window."""
    # Input data
    games_df = pd.DataFrame({
        'home_team': ['Team A', 'Team A', 'Team A'],
        'away_team': ['Team B', 'Team C', 'Team D'],
        'game_date': pd.to_datetime(['2024-07-15', '2024-07-17', '2024-07-20']),
        'game_date_time': pd.to_datetime(['2024-07-15 19:00:00', '2024-07-17 19:00:00', '2024-07-20 19:00:00']),
        'home_score': [5, 3, 7],
        'away_score': [2, 4, 1]
    })
    
    # For Team A: games on 7/15, 7/17, 7/20
    # 7/15: 0 games in previous 7 days
    # 7/17: 1 game in previous 7 days (7/15)  
    # 7/20: 2 games in previous 7 days (7/15, 7/17)
    expected = pd.Series([0, 0, 0, 1, 1, 1], dtype='float64')
    
    result = team_games_previous_7days(games_df)
    print(result)
    pd.testing.assert_series_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

if __name__=='__main__':
    # Command line: pytest tests/test_features.py
    pytest.main([__file__, "-v"])