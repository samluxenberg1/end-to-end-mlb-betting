import pytest
import pandas as pd
from features.team_stats_features import (
    rolling_avg_team_stat,
    lag_team_stat,
    create_team_stats_features
)

def test_rolling_avg_team_stat():
    """
    Test that rolling_avg_team_stat correctly calculates 
    rolling averages with data leakage prevention.
    """
    # Input data
    team_stats_df = pd.DataFrame({
        'team_id': [108,108,108,108,108],
        'season': [2024,2024,2024,2024,2024],
        'game_date_time': pd.to_datetime([
            '2024-04-01 19:05:00', '2024-04-02 19:05:00', '2024-04-03 19:05:00', 
            '2024-04-04 19:05:00', '2024-04-05 19:05:00'
        ]),
        'obp': [0.300, 0.250, 0.400, 0.350, 0.200]
    })

    # Expected output with 3-game rolling average (closed='left' excludes current game)
    expected = pd.Series([
        0.000, # Game 1: no prior games, min_periods=1 --> 0 (fillna)
        0.300, # Game 2: avg of [0.300] = 0.300
        0.275, # Game 3: avg of [0.300, 0.250] = 0.275
        0.317, # Game 4: avg of [0.300, 0.250, 0.400] = 0.317 (rounded)
        0.333  # Game 5: avg of [0.250, 0.400, 0.350] = 0.333 (rounded)
    ])

    result = rolling_avg_team_stat(team_stats_df, 'obp', window=3)

    # Use approximate equality for floating point comparison
    pd.testing.assert_series_equal(
        result, 
        expected, 
        check_names=False, 
        check_exact=False, 
        rtol=1e-3, 
        atol=1e-3
        )

def test_rolling_avg_team_stat_multiple_teams():
    """Test rolling averages work correctly across multiple teams and seasons."""
    # Input data - two teams, two seasons
    team_stats_df = pd.DataFrame({
        'team_id': [108, 108, 109, 109, 108, 109],
        'season': [2023, 2023, 2023, 2023, 2024, 2024], 
        'obp': [0.300, 0.400, 0.250, 0.350, 0.320, 0.280]
    })
    
    # Expected: rolling averages should reset for each team-season combination - groups and therefore sorts
    expected = pd.Series([
        0.000,  # Team 108, 2023, Game 1: no prior games
        0.300,  # Team 108, 2023, Game 2: avg of [0.300]
        0.000,  # Team 108, 2024, Game 1: no prior games (new season)
        0.000,  # Team 109, 2023, Game 1: no prior games  
        0.250,  # Team 109, 2023, Game 2: avg of [0.250]
        0.000   # Team 109, 2024, Game 1: no prior games (new season)
    ])
    
    result = rolling_avg_team_stat(team_stats_df, 'obp', window=2)
    print(team_stats_df)
    print(result)
    pd.testing.assert_series_equal(
        result, 
        expected, 
        check_exact=False, 
        check_names=False, 
        rtol=1e-3, 
        atol=1e-3
        )
    
def test_rolling_avg_data_leakage_prevention():
    """Test that rolling averages with closed='left' prevent data leakage."""
    # Simple test case to verify current game is excluded
    team_stats_df = pd.DataFrame({
        'team_id': [108, 108, 108],
        'season': [2024, 2024, 2024],
        'obp': [0.200, 0.400, 0.600]  # Increasing values
    })
    
    result = rolling_avg_team_stat(team_stats_df, 'obp', window=2)
    
    # Game 3 rolling avg should be avg of [0.200, 0.400] = 0.300
    # NOT avg of [0.400, 0.600] = 0.500 (which would include current game)
    assert result.iloc[2] == pytest.approx(0.300), f"Expected 0.300, got {result.iloc[2]}"

def test_lag_team_stat():
        """Test basic lagging functionality with single team and season."""
        # Arrange
        df = pd.DataFrame({
            'team_id': ['108', '108', '108', '108'],
            'season': [2023, 2023, 2023, 2023],
            'batting_avg': [0.250, 0.260, 0.270, 0.280]
        })
        
        # Act
        result = lag_team_stat(df, 'batting_avg', lag=1)
        
        # Assert
        expected = pd.Series([0.0, 0.250, 0.260, 0.270])
        pd.testing.assert_series_equal(result, expected, check_names=False, check_exact=False, atol=1e-10)

# Should probably add test for create_team_stats_features()....
    

if __name__=='__main__':
    # Command line: pytest tests/test_team_stats_features.py
    pytest.main([__file__, "-v"])