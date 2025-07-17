import requests

def load_games_for_date(date_str):
    """
    Fetch list of scheduled MLB games for a given date.

    Args:
        date_str (str): Date in YYYY-MM-DD format.

    Returns:
        List of dicts: Each dict includes game_pk, game_date, home_team, away_team, venue, status.
    """
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    games = []
    for date in data.get('dates', []):
        for game in date.get('games', []):
            games.append({
                'game_pk': game['gamePk'],
                'game_date': game['gameDate'],
                'home_team': game['teams']['home']['team']['name'],
                'away_team': game['teams']['away']['team']['name'],
                'venue': game['venue']['name'],
                'status': game['status']['detailedState'], # statis = "Scheduled" means game hasn't started yet... (i think -- need to verify)
            })
    return games


if __name__ == '__main__':
    games = load_games_for_date("2025-07-07")
    for g in games:
        print(g)