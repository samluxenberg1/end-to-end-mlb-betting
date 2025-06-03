import requests
from datetime import date
from typing import List

def fetch_games(date_str: str) -> List[dict]:
    """
    Scrapes MLB games from statsapi.mlb.com given a specified date. 

    Args
        date_str: date for which to fetch games from the schedule
    
    Returns
        List of game dictionaries
    """
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    games = resp.json()["dates"][0]["games"]

    return [
        {
            "game_id": str(g["gamePk"]),
            "game_date": date_str,
            "home_team": g["teams"]["home"]["team"]["name"],
            "away_team": g["teams"]["away"]["team"]["name"],
            "home_score": g["teams"]["home"]["score"],
            "away_score": g["teams"]["away"]["score"],
        }
        for g in games
    ]

if __name__ == "__main__":
    print(fetch_games(date.today().isoformat())[:2])  # quick sanity check
