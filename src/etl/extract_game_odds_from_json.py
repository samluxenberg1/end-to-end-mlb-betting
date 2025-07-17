import json
from datetime import datetime
import logging

PREFERRED_BOOKMAKERS = ["betmgm", "draftkings", "fanduel", "williamhill_us", "espnbet"]

def parse_odds_json(json_path: str) -> list[dict]:
    # Read in the file
    with open(json_path, "r") as f:
        json_data = json.load(f)
    
    games = json_data.get("data", [])
    parsed_games = []

    for game in games:
        home_team = game.get("home_team")
        away_team = game.get("away_team")
        commence_time = game.get("commence_time")

        dt_commence_time = datetime.strptime(commence_time, "%Y-%m-%dT%H:%M:%SZ")
        game_date = datetime.strftime(dt_commence_time, "%Y-%m-%d")

        selected_market = None
        selected_bookmaker = None

        # Try Preferred Bookmakers First
        for book in game.get("bookmakers", []):
            if book.get("key") in PREFERRED_BOOKMAKERS:
                for market in book.get("markets", []):
                    if market.get("key") == 'h2h':
                        selected_bookmaker = book.get("key")
                        selected_market = market
                        break
                if selected_market:
                    break

        # Fallback: try any bookmaker with h2h
        if not selected_market:
            for book in game.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market.get("key") == 'h2h':
                        selected_bookmaker = book.get("key")
                        selected_market = market
                        break
                if selected_market:
                    break

        if selected_market:
            outcomes = {o.get("name"): o.get("price") for o in selected_market.get("outcomes", [])}
            
            parsed_games.append({
                "game_date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": commence_time,
                "bookmaker": selected_bookmaker,
                "home_odds": outcomes.get(home_team),
                "away_odds": outcomes.get(away_team)
            })

    return parsed_games


if __name__ == '__main__':
    print(parse_odds_json("data/odds_api_raw/2021-04-01.json"))

    import glob
    import os
    import csv
    

    input_dir = "data/odds_api_raw"
    output_csv = "data/processed_game_dds"

    all_parsed_game_odds = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_file = os.path.join(input_dir, filename)
            parsed_game_odds = parse_odds_json(input_file)
            all_parsed_game_odds.extend(parsed_game_odds)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_parsed_game_odds[0].keys())
        writer.writeheader()
        writer.writerows(all_parsed_game_odds)

    logging.info(f"Saved {len(all_parsed_game_odds)} game odds to {output_csv}")