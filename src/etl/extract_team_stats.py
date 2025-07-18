import argparse
import logging
from datetime import datetime, timedelta
from utils import fetch_team_stats, fetch_games_for_date
import csv
import os

def daterange(start_date: datetime, end_date: datetime):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)

def extract_team_stats(start_str, end_str, output_csv="data/raw_team_stats.csv"):
    """
    Extract team stats for each game for a date range and save to CSV.
    
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
    all_team_stats = []
    for single_date in daterange(start_date, end_date):
        date_str = single_date.strftime("%Y-%m-%d")

        logging.info(f"Fetching games for {date_str}")
        try: 
            games = fetch_games_for_date(date_str, max_retries=3)    
        except Exception as e:
            logging.error(f"Failed to fetch games for {date_str}: {e}")
            continue
        
        game_ids = [g["game_id"] for g in games]
        
        # For each game, fetch team statistics
        for game_id in game_ids:
            try:
                team_stats = fetch_team_stats(game_id, max_retries=3)
                if team_stats:
                    all_team_stats.extend(team_stats)
            except Exception as e:
                logging.warning(f"Failed to fetch team stats for game {game_id} on {date_str}: {e}")
                continue

    # Save to CSV
    os.makedirs("data", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_team_stats[0].model_dump().keys())
        writer.writeheader()
        writer.writerows([team_stat.model_dump() for team_stat in all_team_stats])

    logging.info(f"Saved {len(all_team_stats)} teams to {output_csv}")

    return all_team_stats

if __name__=="__main__":
    # For command line execution: python etl/extract_team_stats.py --start-date yyyy-mm-dd --end-date yyyy-mm-dd
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="data/raw_team_stats.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    extract_team_stats(args.start_date, args.end_date, args.output_csv)