import logging
from datetime import date, datetime, timedelta

from src.etl.utils import fetch_latest_game_date, fetch_games, fetch_team_stats, fetch_player_stats
from src.etl.load_to_db import load_games_to_db, load_team_stats_to_db, load_player_stats_to_db

DB_CONFIG = {
    "dbname": "mlb_db",
    "user": "mlb_user", 
    "password": "mlb_pass",
    "host": "localhost",
    "port": 5432,
    }

# Get latest game date in DB
def daterange(start_date: date, end_date: date):
    current = start_date

    while current <= end_date:
        yield current
        current += timedelta(days=1)

def update_all_data(db_config: dict, max_retries: int):
    """
    Fetches the latest game, team, and player data for all games after latest game date in games table in DB.

    Args
        db_cong: database configuration dictionary
        max_retries: number of retry requests when fetching from MLB Stats API
    """
    logging.basicConfig(level=logging.INFO)
    
    # Find date to start fetching new data
    today = date.today()
    latest_date = fetch_latest_game_date(db_config)

    logging.info(f"Fetching new data from {latest_date-timedelta(days=10)} to {today}")

    all_games, all_team_stats, all_player_stats = [], [], []

    # For each date range from latest date in games table until today,
    # fetch all games and team/player stats from each of those games.
    for single_date in daterange(latest_date-timedelta(days=10), today):
        single_date_str = single_date.strftime("%Y-%m-%d")
        try:
            logging.info(f"Fetching game data for {single_date_str}...")
            games = fetch_games(date_str=single_date_str, max_retries=max_retries)
            all_games.extend(games)

            for game in games:
                game_id = game["game_id"]

                try:
                    team_stats = fetch_team_stats(game_pk=game_id, max_retries=max_retries)
                    if team_stats:
                        all_team_stats.extend(team_stats)
                except Exception as e:
                    logging.warning(f"Team stats failed for game {game_id}: {e}")
                
                try:
                    player_stats = fetch_player_stats(game_pk=game_id, max_retries=max_retries)
                    if player_stats:
                        all_player_stats.extend(player_stats)
                except Exception as e:
                    logging.warning(f"Player stats failed for game {game_id}: {e}")
        
        except Exception as e:
            logging.error(f"Failed to fetch games for {single_date_str}: {e}")

    # Load to DB
    if all_games: 
        load_games_to_db(data=all_games, db_config=db_config, from_memory=True)
    if all_team_stats:
        load_team_stats_to_db(data=all_team_stats, db_config=db_config, from_memory=True)
    if all_player_stats:
        load_player_stats_to_db(data=all_player_stats, db_config=db_config, from_memory=True)

if __name__=="__main__":
    update_all_data(DB_CONFIG, 3)

    # test game_id/gamepk = 777146 (july 11, 2025)



