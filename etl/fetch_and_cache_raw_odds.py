import os
import json
import time
import logging
import requests
from datetime import datetime
from gen_utils_etl import daterange

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("ODDS_API_KEY")
HIST_URL = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds"


def get_final_pregame_odds(bookmakers, commence_time_str):
    commence_time_dt = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))

def fetch_raw_odds_for_date(date_str: str): 
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "date": f"{date_str}T16:00:00Z" # 12 pm eastern (4pm UTC)
    }
    response = requests.get(HIST_URL, params=params, timeout=10)
    response.raise_for_status()
    return response.json()

def save_odds_for_date(date_str: str, output_dir: str = "data/odds_api_raw"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{date_str}.json")
    if os.path.exists(output_path):
        logging.info(f"File already exists for {date_str}, skipping...")
        return 
    try:
        odds_data = fetch_raw_odds_for_date(date_str)
        with open(output_path, "w") as f:
            json.dump(odds_data, f)
        logging.info(f"Saved odds for {date_str} to {output_path}")
        time.sleep(1.1)
    except Exception as e:
        logging.warning(f"Failed to fetch odds for {date_str}: {e}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Add a check to verify API key is loaded
    if not API_KEY:
        logging.error("ODDS_API_KEY environment variable is not set!")
        exit(1)
    
    logging.info(f"Starting odds fetch from {args.start_date} to {args.end_date}")
    logging.info(f"API Key loaded: {'Yes' if API_KEY else 'No'}")

    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    for d in daterange(start, end):
        date_str = d.strftime("%Y-%m-%d")
        save_odds_for_date(date_str)