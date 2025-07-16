import pandas as pd
from features.utils_players import (
    STAR_BATTERS, 
    STAR_PITCHERS,
    get_star_players,
    count_missing_star_players,
    get_players_in_game
)
# Features
# Lineup_avg_obp - find players who played in the current game and compute their avg obp PRIOR to the current game
# num_batters_with_pa - depth proxy
# num_star_players_missing - based on shortlist
# num_expected_starters_missing - seems a little too complicated for this first try
# starting_pitcher_absent - useful for games without starter data
# bullpen_game_indicator - inferred via no pitcher with IP > 3

# NEED TO COME BACK TO THIS>>>
def add_missing_star_players_features(df_games: pd.DataFrame, df_player_stats: pd.DataFrame) -> pd.DataFrame:
    df_games['season'] = pd.to_datetime(df_games['game_date']).dt.year
    
    for side in ['home', 'away']:
        df_games[f"{side}_star_batters_missing"] = df_games.apply(
            lambda row: count_missing_star_players(
                df_player_stats,
                row['game_id'],
                row["f{side}_team_id"]
            ),
            row[f"{side}_team"],
            row["season"]
        )


if __name__=='__main__': 
