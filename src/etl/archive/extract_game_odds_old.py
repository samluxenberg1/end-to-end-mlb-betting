import argparse
import logging
from datetime import datetime, timedelta
from utils_odds import fetch_final_odds_for_game
from gen_utils_etl import daterange
from utils import fetch_games
import csv
import os

def extract_game_odds(start_str: str, end_str: str, output_csv="data/raw_game_odds.csv"):
    # Convert string dates to datetime
    start_date = datetime.strptime(start_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_str, "%Y-%m-%d")

    # Iterate through all dates in date range and fetch each day's odds
    all_game_odds = []
    for single_date in daterange(start_date, end_date):
        date_str = datetime.strftime(single_date, "%Y-%m-%d")

        logging.info(f"Fetch games for {date_str}")
        try: 
            games = fetch_games(date_str, max_retries=3)
        except Exception as e:
            logging.error(f"Failed to fetch games for {date_str}: {e}")
            continue

        game_info = [(g['game_date'],g['home_team'], g['away_team']) for g in games]

        # For each game fetch odds
        for game_date, home_team, away_team in game_info:
            try:
                game_odds = fetch_final_odds_for_game(game_date, home_team, away_team)
                if game_odds:
                    # Add game information for joining purposes later
                    game_odds.update({
                        "game_date": game_date,
                        "home_team": home_team,
                        "away_team": away_team
                    })
                    all_game_odds.append(game_odds)
            except Exception as e: 
                logging.warning(f"Failed to fetch odds for {game_date}: {home_team} vs. {away_team}: {e}")
                continue
    
    if not all_game_odds:
        logging.warning("No odds data fetched. CSV not written.")
        return
    
    # Save to CSV
    os.makedirs("data", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_game_odds[0].keys())
        writer.writeheader()
        writer.writerows(all_game_odds)
    logging.info(f"Saved {len(all_game_odds)} odds to {output_csv}")

    return all_game_odds

if __name__ == '__main__':
    # For command line execution: python etl/extract_game_odds.py --start-date yyyy-mm-dd --end-date yyyy-mm-dd
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="data/raw_game_odds.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    extract_game_odds(args.start_date, args.end_date, args.output_csv)
        