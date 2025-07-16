import pytest
import pandas as pd
from features.games_features import (
    team_schedule,
    team_rest_days,
    team_games_previous_7days, 
    team_win_rate_last_10,
    merge_team_features_into_games,
    create_game_features
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
        'season': [2024, 2024, 2024, 2024],
        'team_season_opener_flag': [1,0,1,1]
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
    print(games_df)
    
    # For Team A: games on 7/15, 7/17, 7/20
    # 7/15: 0 games in previous 7 days
    # 7/17: 1 game in previous 7 days (7/15)  
    # 7/20: 2 games in previous 7 days (7/15, 7/17)
    # For Team B: games on 7/15
    # 7/15: 0 games in previous 7 days
    # For Team C: game on 7/17
    # 7/17: 0 games in previous 7 days
    # For Team D: game on 7/20
    # 7/20: 0 games in previous 7 days
    expected = pd.Series([0, 1, 2, 0, 0, 0], dtype='float64')
    
    result = team_games_previous_7days(games_df)
    print(result)
    pd.testing.assert_series_equal(
        result.reset_index(drop=True), 
        expected.reset_index(drop=True),
        check_names=False
    )
    
def test_team_win_rate_last_10():
    """Test that team_win_rate_last_10 correctly calculates rolling win rate."""
    # Input data - Team A wins first game, loses second
    games_df = pd.DataFrame({
        'home_team': ['Team A', 'Team B'],
        'away_team': ['Team B', 'Team A'],
        'game_date': pd.to_datetime(['2024-07-15', '2024-07-16']),
        'game_date_time': pd.to_datetime(['2024-07-15 19:00:00', '2024-07-16 19:00:00']),
        'home_score': [5, 2],
        'away_score': [3, 4]
    })
    
    # Team A: wins 7/15 (home), wins 7/16 (away) -> win rates: 0, 1
    # Team B: loses 7/15 (away), loses 7/16 (home) -> win rates: 0, 0
    expected = pd.Series([0.0, 1.0, 0.0, 0.0], dtype='float64')
    
    result = team_win_rate_last_10(games_df)
    pd.testing.assert_series_equal(
        result.reset_index(drop=True), 
        expected.reset_index(drop=True),
        check_names=False
    )

def test_merge_team_features_into_games():
    """Test that merge_team_features_into_games correctly merges team features."""
    # Original games DataFrame
    df_games = pd.DataFrame({
        'home_team': ['Team A', 'Team B'],
        'away_team': ['Team B', 'Team A'],
        'game_date': pd.to_datetime(['2024-07-15','2024-07-16']),
        'game_date_time': pd.to_datetime(['2024-07-15 19:00:00', '2024-07-16 19:00:00']),
        'home_score': [5, 3],
        'away_score': [2, 4]
    })
    
    # Team schedule with features
    df_team_schedule = team_schedule(df_games) # previously tested
    df_team_schedule['team_rest_days'] = team_rest_days(df_games) # previously tested
    df_team_schedule['team_games_prev_7days'] = team_games_previous_7days(df_games) # previously tested
    
    # Expected output
    expected = pd.DataFrame({
        'home_team': ['Team A', 'Team B'],
        'away_team': ['Team B', 'Team A'],
        'game_date': pd.to_datetime(['2024-07-15','2024-07-16']),
        'game_date_time': pd.to_datetime(['2024-07-15 19:00:00', '2024-07-16 19:00:00']),
        'home_score': [5, 3],
        'away_score': [2, 4],
        'home_team_rest_days': [0.0, 0.0],
        'home_team_games_prev_7days': [0.0, 1.0],
        'away_team_rest_days': [0.0, 0.0],
        'away_team_games_prev_7days': [0.0, 1.0]
    })
    
    team_schedule_cols = ['team', 'team_rest_days', 'team_games_prev_7days']
    result = merge_team_features_into_games(df_games, df_team_schedule, team_schedule_cols)
    
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

def test_create_game_features():
    """Test that create_game_features correctly creates all feature categories."""
    # Input data - simple 2-game scenario
    games_df = pd.DataFrame({
        'home_team': ['Team A', 'Team B'],
        'away_team': ['Team B', 'Team A'],
        'game_date': pd.to_datetime(['2024-07-15', '2024-07-17']),
        'game_date_time': pd.to_datetime(['2024-07-15 19:00:00', '2024-07-17 13:30:00']),
        'home_score': [5, 2],
        'away_score': [3, 6]
    })
    
    # Expected output with all features
    expected_df = pd.DataFrame({
        'home_team': ['Team A', 'Team B'],
        'away_team': ['Team B', 'Team A'],
        'game_date': pd.to_datetime(['2024-07-15', '2024-07-17']),
        'game_date_time': pd.to_datetime(['2024-07-15 19:00:00', '2024-07-17 13:30:00']),
        'home_score': [5, 2],
        'away_score': [3, 6],
        
        # Schedule Features
        'home_team_rest_days': [0, 1],  # First games for each team
        'home_team_games_prev_7days': [0, 1], 
        'home_team_season_opener_flag': [1, 0],
        'away_team_rest_days': [0, 1],  # Team B: 0, Team A: 1 day rest
        'away_team_games_prev_7days': [0, 1],
        'away_team_season_opener_flag': [1, 0],
        'home_back2back': [0, 0],  
        'away_back2back': [0, 0],
        'rest_difference': [0, 0],  # home_rest - away_rest
        
        # Outcome Features (all 0 for first games)
        'home_win_rate_last_10': [0.0, 0.0],
        'home_avg_run_diff_last_10': [0.0, -2.0],
        'home_std_run_diff_last_10': [0.0, 0.0],
        'home_avg_runs_scored_last_10': [0.0, 3.0],  # Team B scored 3 in first game
        'home_std_runs_scored_last_10': [0.0, 0.0],
        'home_avg_runs_allowed_last_10': [0.0, 5.0],  # Team B allowed 5 runs in first game
        'home_std_runs_allowed_last_10': [0.0, 0.0],
        'away_win_rate_last_10': [0.0, 1.0],  # Team A won first game
        'away_avg_run_diff_last_10': [0.0, 2.0],  # Team A won by 2 in first game
        'away_std_run_diff_last_10': [0.0, 0.0],
        'away_avg_runs_scored_last_10': [0.0, 5.0],  # Team A scored 5 in first game
        'away_std_runs_scored_last_10': [0.0, 0.0],
        'away_avg_runs_allowed_last_10': [0.0, 3.0],  # Team A allowed 3 in first game
        'away_std_runs_allowed_last_10': [0.0, 0.0],
        
        # Temporal Features
        'game_month': [7, 7],  # July
        'game_day_of_week': [0, 2],  # Monday, Wednesday
        'game_hour': [19, 13]  # 7 PM, 1:30 PM
    })
    
    result = create_game_features(games_df)
    
    # Test the complete DataFrame
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected_df.reset_index(drop=True),
        check_dtype=False  # Allow for minor dtype differences
    )

if __name__=='__main__':
    # Command line: pytest tests/test_games_features.py
    pytest.main([__file__, "-v"])