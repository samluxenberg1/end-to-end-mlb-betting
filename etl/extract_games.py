import argparse
import logging
from datetime import datetime, timedelta
from utils import fetch_games
import csv
import os

def daterange(start_date: datetime, end_date: datetime):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)

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
    # Convert string dates to datetime
    start_date = datetime.strptime(start_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_str, "%Y-%m-%d")

    # Iterate through all dates in date range and fetch each day's game information
    all_games = []
    for single_date in daterange(start_date, end_date):
        date_str = single_date.strftime("%Y-%m-%d")
        try:
            logging.info(f"Fetching games for {date_str}")
            games = fetch_games(date_str, max_retries=3)
            logging.info(f" --> Found{len(games)} games")
            all_games.extend(games)

        except Exception as e:
            logging.warning(f"Failed to fetch games for {date_str}: {e}")
    
    if not all_games:
        logging.warning("No games fetched. CSV file will not be written.")
        return []

    # Save to CSV
    os.makedirs("data", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_games[0].keys())
        writer.writeheader()
        writer.writerows(all_games)

    logging.info(f"Saved {len(all_games)} games to {output_csv}")

    return all_games

if __name__=="__main__":
    # For command line execution: python etl/extract_games.py --start-date yyyy-mm-dd --end-date yyyy-mm-dd
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="data/raw_games.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    extract_games(args.start_date, args.end_date, args.output_csv)