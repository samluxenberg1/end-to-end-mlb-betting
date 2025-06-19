import os
import requests
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("ODDS_API_KEY")
HIST_URL = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds"

def fetch_final_odds_for_game(game_date_str: str, home_team: str, away_team: str):
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "date": f"{game_date_str}T23:59:59z" # fetch closest snap shot at end-of-day
    }
    resp = requests.get(HIST_URL, params=params, timeout=10)
    resp.raise_for_status()

    for game in resp.json().get("data", []):
        if game["home_team"] == home_team and game["away_team"] == away_team:
            # Pick BetMGM Bookmaker
            # Find BetMGM bookmaker in the list
            book = next((b for b in game["bookmakers"] if b["key"] == "betmgm"), None)
            if not book:
                print(f"No BetMGM odds found for {home_team} vs {away_team}")
                continue
            m = next((m for m in book["markets"] if m["key"] == "h2h"), None)
            if m:
                odds = {o["name"]: o["price"] for o in m["outcomes"]}
                return {
                    "home": odds.get(home_team),
                    "away": odds.get(away_team),
                    "bookmaker": book["key"],
                    "market": m["key"],
                    "commence_time": game["commence_time"]
                }
    return None


if __name__=="__main__":
    result = fetch_final_odds_for_game(game_date_str='2021-04-01', home_team='San Diego Padres', away_team='Arizona Diamondbacks')
    print(result)