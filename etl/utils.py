import requests
from datetime import date

def fetch_games_for_date(date_str: str) -> list[dict]:
    """
    Scrapes MLB game ids from statsapi.mlb.com given a specified date.

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
            "game_date": date_str
        }
        for g in games
    ]

def fetch_games(date_str: str) -> list[dict]:
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
            "state": g["status"]["detailedState"],
            "venue": g.get("venue", {}).get("name", ""),
            "game_type": g.get("gameType", "")
        }
        for g in games
    ]

def fetch_team_stats(game_pk: str) -> list[str]:
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status() # checks if http request was successful
    data = resp.json()

    teams = ['home', 'away']
    rows = []

    for side in teams:
        team_data = data['teams'][side]
        team_id = team_data['team']['id']
        stats_batting = team_data['teamStats']['batting']
        stats_pitching = team_data['teamStats']['pitching']
        stats_fielding = team_data['teamStats']['fielding']

        row = {
            "game_pk": game_pk,
            "team_side": side,
            "team_id": team_id,
            "runs_batting": stats_batting.get('runs', 0),
            "hits_batting": stats_batting.get('hits', 0),
            "strikeOuts_batting": stats_batting.get('strikeOuts', 0),
            "baseOnBalls_batting": stats_batting.get('baseOnBalls', 0),
            "avg": stats_batting.get('avg', 0),
            "obp": stats_batting.get('obp', 0),
            "slg": stats_batting.get('slg', 0),
            "pitchesThrown": stats_pitching.get('pitchesThrown', 0),
            "balls_pitching": stats_pitching.get('balls', 0),
            "strikes_pitching": stats_pitching.get('strikes', 0),
            "strikeOuts_pitching": stats_pitching.get('strikeOuts', 0),
            "baseOnBalls_pitching": stats_pitching.get('baseOnBalls', 0),
            "hits_pitching": stats_pitching.get('hits', 0),
            "earnedRuns": stats_pitching.get('earnedRuns', 0),
            "homeRuns_pitcing": stats_pitching.get('homeRuns', 0),
            "runs_pitching": stats_pitching.get('runs', 0),
            "era": stats_pitching.get('era', 0),
            "whip": stats_pitching.get('whip', 0),
            "groundOuts_pitching": stats_pitching.get('groundOuts', 0),
            "airOuts_pitching": stats_pitching.get('airOuts', 0),
            "total": stats_fielding.get('total', 0),
            "putOuts": stats_fielding.get('putOuts', 0),
            "assists": stats_fielding.get('assists', 0),
            "errors": stats_fielding.get('errors', 0),
            "doublePlays": stats_fielding.get('doublePlays', 0),
            "triplePlays": stats_fielding.get('triplePlays', 0),
            "rangeFactor": stats_fielding.get('rangeFactor', 0),
            "caughtStealing": stats_fielding.get('caughtStealing', 0),
            "passedBall": stats_fielding.get('passedBall', 0),
            "innings": stats_fielding.get('innings', 0),
        }

        rows.append(row)

    return rows



if __name__ == "__main__":
    #print(fetch_games(date.today().isoformat())[:2])  # quick sanity check
    #print("\n\n")
    #print(fetch_team_stats(634627))
    print("\n")
    print(fetch_games_for_date(date.today().isoformat()))

