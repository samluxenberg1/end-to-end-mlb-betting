import os
import requests
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("ODDS_API_KEY")
HIST_URL = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds"

def fetch_final_odds_for_game(game_date_str: str, home_team: str, away_team: str):
    
    preferred_bookmakers = ["betmgm", "draftkings", "fanduel", "williamhill_us", "espnbet"]
    
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
        if game.get("home_team") == home_team and game.get("away_team") == away_team:
            # Try bookmakers in order of preference
            for bookmaker_key in preferred_bookmakers:
                book = next((b for b in game.get("bookmakers", []) if b.get("key") == bookmaker_key), None)
                if not book:
                    continue
                market = next((m for m in book.get("markets", []) if m.get("key") == "h2h"), None)
                if not market:
                    continue
                odds = {o.get("name"): o.get("price") for o in market.get("outcomes", [])}
                return {
                    "home": odds.get(home_team),
                    "away": odds.get(away_team),
                    "bookmaker": book.get("key"),
                    "market": market.get("key"),
                    "commence_time": game.get("commence_time")
                }
            
            # If no preferred bookmakers found, try any available bookmaker
            for book in game.get("bookmakers",[]):
                market = next((m for m in book.get("markets", []) if m.get("key") == "h2h"), None)
                if market: 
                    odds = {o.get("name"): o.get("price") for outcome in market.get("outcomes", [])}
                    return {
                        "home": odds.get(home_team),
                        "away": odds.get(away_team),
                        "bookmaker": book.get("key"),
                        "market": market.get("key"),
                        "commence_time": game.get("commence_time")
                    }
            
            # If we get here, no bookmaker has odds for this game
            print(f"No bookmaker has odds for {game_date_str}: {home_team} vs. {away_team}")
            return None
    
    print(f"Game not found for {game_date_str}: {home_team} vs. {away_team}")
    return None


if __name__=="__main__":
    result = fetch_final_odds_for_game(game_date_str='2021-04-01', home_team='San Diego Padres', away_team='Arizona Diamondbacks')
    print(result)
    