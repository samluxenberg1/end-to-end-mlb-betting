import pytest
import pandas as pd
from features.games_features import team_schedule

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
        'team': ['Los Angeles Dodgers', 'Boston Red Sox', 'Boston Red Sox', 'New York Yankees'],
        'game_date': ['2024-07-16', '2024-07-15', '2024-07-16', '2024-07-15'],
        'game_date_time': ['2024-07-16 19:10:00', '2024-07-15 19:05:00', '2024-07-16 19:10:00', '2024-07-15 19:05:00'],
        'home_ind': [0, 0, 1, 1]
    })
    
    result = team_schedule(games_df)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

    if __name__=='__main__':
        test_team_schedule()