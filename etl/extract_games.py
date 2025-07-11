import argparse
import logging
from datetime import datetime, timedelta
from utils import fetch_games
import csv
import os
import pandas as pd

def daterange(start_date: datetime, end_date: datetime):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)

# def get_latest_game_date(csv_path: str):
#     if not os.path.exists(csv_path):
#         return None

#     latest_date = None
#     df = pd.read_csv(csv_path, parse_dates=['game_date', 'game_date_time'])
#     latest_date = df['game_date'].max().normalize()

#     return latest_date

def extract_games(start_str, end_str, output_csv="data/raw_games.csv"):
    """
    Extract game data for a date range and save to CSV.
    
    Args:
        start_str (str): Start date in YYYY-MM-DD format.
        end_str (str): End date in YYYY-MM-DD format.
        output_csv (str): Output CSV file path.
    
    Returns:
        list: List of game dictionaries.
    """
    # Set up
    all_new_games = []
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)

    # Load existing data if it exists
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv, parse_dates=['game_date', 'game_date_time'])
        existing_game_ids = set(existing_df['game_id'])
        latest_saved_date = existing_df['game_date'].max()
        logging.info(f"Loaded {len(existing_game_ids)} games (latest date: {latest_saved_date})")
    else:
        existing_df = pd.DataFrame()
        existing_game_ids = set()
        latest_saved_date = None
        logging.info("No existing game data found. Will fetch all games from scratch.")

    # Convert string dates to datetime
    start_date = datetime.strptime(start_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_str, "%Y-%m-%d")

    # If existing data found, only re-fetch today's date
    if latest_saved_date and latest_saved_date > start_date:
        start_date = latest_saved_date
    

    # Iterate through all dates in date range and fetch each day's game information
    all_games = []
    for single_date in daterange(start_date, end_date):
        date_str = single_date.strftime("%Y-%m-%d")
        try:
            logging.info(f"Fetching games for {date_str}")
            games = fetch_games(date_str, max_retries=3)
            logging.info(f" --> Found {len(games)} new games")
            all_games.extend(games)

        except Exception as e:
            logging.warning(f"Failed to fetch games for {date_str}: {e}")
    
    if not all_games:
        logging.warning("No games fetched. CSV file will not be written.")
        return []

    # Collect new games into dataframe
    new_games_df = pd.DataFrame(all_games)

    # Append to CSV
    new_games_df.to_csv(output_csv, model='a', index=False, header= not os.path.exists(output_csv))
    logging.info(f"Appended {len(new_games_df)} new games to {output_csv}")

    # Save to CSV
    # os.makedirs("data", exist_ok=True)
    # with open(output_csv, "w", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=all_games[0].keys())
    #     writer.writeheader()
    #     writer.writerows(all_games)

    # logging.info(f"Saved {len(all_games)} games to {output_csv}")

    return new_games_df

if __name__=="__main__":
    # For command line execution: python etl/extract_games.py --start-date yyyy-mm-dd --end-date yyyy-mm-dd
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="data/raw_games.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    extract_games(args.start_date, args.end_date, args.output_csv)