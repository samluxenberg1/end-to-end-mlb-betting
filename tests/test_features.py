import pytest
import pandas as pd
from features.games_features import team_schedule, team_rest_days

def test_team_schedule():
    """Test that team_schedule correctly transforms games into team schedules."""
    # Input data
    games_df = pd.DataFrame({
        'home_team': ['New York Yankees', 'Boston Red Sox'],
        'away_team': ['Boston Red Sox', 'Los Angeles Dodgers'],
        'game_date': ['2024-07-15', '2024-07-16'],
        'game_date_time': ['2024-07-15 19:05:00', '2024-07-16 19:10:00']
    })
    
    # Expected output
    expected = pd.DataFrame({
        'team': ['Boston Red Sox', 'Boston Red Sox', 'Los Angeles Dodgers', 'New York Yankees'],
        'game_date': ['2024-07-15', '2024-07-16', '2024-07-16', '2024-07-15'],
        'game_date_time': ['2024-07-15 19:05:00', '2024-07-16 19:10:00', '2024-07-16 19:10:00', '2024-07-15 19:05:00'],
        'home_ind': [0,1,0,1]
    })
    
    # Result
    result = team_schedule(games_df)
    
    # Test
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

def test_team_rest_days():
    
    games_df = pd.DataFrame({
        'home_team': ['New York Yankees','Boston Red Sox','New York Yankees'],
        'away_team': ['Boston Red Sox','Baltimore Orioles','Toronto Blue Jays'],
        'game_date': pd.to_datetime(['2024-07-15','2024-07-16','2024-07-20']),
        'game_date_time': pd.to_datetime(['2024-07-15 19:05:00', '2024-07-16 19:10:00','2024-07-20 18:00:00'])
    })

    # Expected output
    expected = pd.Series([0,0,0,0,4,0], name='game_date', dtype='float64')
    
    # Result
    result = team_rest_days(games_df)
    
    # Test
    pd.testing.assert_series_equal(result.reset_index(drop=True), expected)

    if __name__=='__main__':
        pytest.main([__file__, "-v"])