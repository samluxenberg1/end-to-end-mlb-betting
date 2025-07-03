import pandas as pd
from features.utils_players import (
    STAR_BATTERS, 
    STAR_PITCHERS,
    get_star_players
)
# Features
# Lineup_avg_obp - find players who played in the current game and compute their avg obp PRIOR to the current game
# num_batters_with_pa - depth proxy
# num_star_players_missing - based on shortlist
# num_expected_starters_missing - seems a little too complicated for this first try
# starting_pitcher_absent - useful for games without starter data
# bullpen_game_indicator - inferred via no pitcher with IP > 3

def count_missing_star_players(df_player_stats, df_games):
    # Join on game_id and team
    df = pd.merge(
        df_games[['game_id','game_date','home_team','away_team']],
        df_player_stats[['game_pk','team_id','player_name']],
        left_on='game_id',
        right_on='game_pk',
        how='left'
    )

    df['game_year'] = pd.to_datetime(df['game_date']).dt.year

    def missing_star_players(row, side='home'):
        team = row[f"{side}_team"]
        year = row["game_year"]
        players = df[
            (df['game_date'] == row['game_id']) & 
            (df['team_id'] == row[f"{side}_team_id"])
        ]['player_name'].dropna().tolist()
        stars = get_star_players(team, year, player_set='????')
        missing = [s for s in stars if s not in players]
        return len(missing)
    
    df_games["home_star_players_missing"] = df_games.apply(lambda row: mising_star_players(row, side='home'),axis=1)
    df_games["away_star_players_missing"] = df_games.apply(lambda row: mising_star_players(row, side='away'),axis=1)

    return df_games

if __name__=='__main__': 
