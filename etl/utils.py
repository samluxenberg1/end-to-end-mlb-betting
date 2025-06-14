import requests
import time
from datetime import date
from pydantic import BaseModel, validator
from typing import Optional
import logging

# Set up logging to track retry attempts
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_request(url: str, max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0, timeout: int = 15) -> requests.Response:
    """
    Make HTTP request with exponential backoff retry logic.
    
    Args:
        url: URL to request
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        timeout: Request timeout in seconds
    
    Returns:
        requests.Response object
    
    Raises:
        requests.RequestException: If all retries fail
    """
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                wait_time = delay * (backoff ** attempt)
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                logger.info(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"All retry attempts failed for URL: {url}")
                raise

def fetch_games_for_date(date_str: str, max_retries: int) -> list[dict]:
    """
    Scrapes MLB game ids from statsapi.mlb.com given a specified date.

    Args
        date_str: date for which to fetch games from the schedule
        max_retries: maximum number of retry attempts

    Returns
        List of game dictionaries
    """
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
    resp = retry_request(url, max_retries=max_retries)
    #resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    games = resp.json()["dates"][0]["games"]
    
    return [
        {
            "game_id": str(g["gamePk"]),
            "game_date": date_str
        }
        for g in games
    ]

def fetch_games(date_str: str, max_retries: int) -> list[dict]:
    """
    Scrapes MLB games from statsapi.mlb.com given a specified date. 

    Args
        date_str: date for which to fetch games from the schedule
        max_retries: maximum number of rety attempts
    
    Returns
        List of game dictionaries
    """
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
    resp = retry_request(url, max_retries=max_retries)
    #resp = requests.get(url, timeout=15)
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

class PlayerStats(BaseModel):
    game_pk: Optional[int] = None
    player_id: Optional[int] = None
    player_name: Optional[str] = None
    team_side: Optional[str] = None  # 'home' or 'away'

    # Pitching stats
    innings_pitched: Optional[float] = None
    hits_allowed: Optional[int] = None
    runs_allowed: Optional[int] = None
    earned_runs: Optional[int] = None
    strikeouts_pitching: Optional[int] = None
    walks_pitching: Optional[int] = None
    pitches_thrown: Optional[int] = None

    # Batting stats
    at_bats: Optional[int] = None
    runs_scored: Optional[int] = None
    hits: Optional[int] = None
    home_runs: Optional[int] = None
    rbis: Optional[int] = None
    walks_batting: Optional[int] = None
    strikeouts_batting: Optional[int] = None
    left_on_base: Optional[int] = None
    stolen_bases: Optional[int] = None

    # Fielding stats
    putouts: Optional[int] = None
    assists: Optional[int] = None
    errors: Optional[int] = None

    @validator('*', pre=True)
    def clean_all_fields(cls, v):
        problem_values = {
            '.---','-.--', '--', '-', '', 
            'N/A', 'n/a', 'NA', 'na',
            'NULL', 'null', 'Null',
            'undefined', 'UNDEFINED',
            None
        }
        if v in problem_values:
            return None
        if isinstance(v, str) and v.strip() == '':
            return None
        return v

class TeamStats(BaseModel):
    # All fields as Optional to handle None values gracefully
    game_pk: Optional[int] = None
    team_side: Optional[str] = None
    team_id: Optional[int] = None
    runs_batting: Optional[int] = None
    hits_batting: Optional[int] = None
    strikeOuts_batting: Optional[int] = None
    baseOnBalls_batting: Optional[int] = None
    avg: Optional[float] = None
    obp: Optional[float] = None
    slg: Optional[float] = None
    pitchesThrown: Optional[int] = None
    balls_pitching: Optional[int] = None
    strikes_pitching: Optional[int] = None
    strikeOuts_pitching: Optional[int] = None
    baseOnBalls_pitching: Optional[int] = None
    hits_pitching: Optional[int] = None
    earnedRuns: Optional[int] = None
    homeRuns_pitching: Optional[int] = None
    runs_pitching: Optional[int] = None
    era: Optional[float] = None
    whip: Optional[float] = None
    groundOuts_pitching: Optional[int] = None
    airOuts_pitching: Optional[int] = None
    total: Optional[int] = None
    putOuts: Optional[int] = None
    assists: Optional[int] = None
    errors: Optional[int] = None
    doublePlays: Optional[int] = None
    triplePlays: Optional[int] = None
    rangeFactor: Optional[float] = None
    caughtStealing: Optional[int] = None
    passedBall: Optional[int] = None
    innings: Optional[float] = None

    @validator('*', pre=True)
    def clean_all_fields(cls, v): 
        """Convert any problematic values to None"""

        problem_values = {
            '-.--', '--', '-', '', 
            'N/A', 'n/a', 'NA', 'na',
            'NULL', 'null', 'Null',
            'undefined', 'UNDEFINED',
            None
        }

        if v in problem_values:
            return None
        
        # If it's a string, make sure it's not whitespace
        if isinstance(v, str) and v.strip() == '':
            return None
        
        return v

def fetch_team_stats(game_pk: str, max_retries: int) -> list[TeamStats]:
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    resp = retry_request(url, max_retries=max_retries)
    #resp = requests.get(url, timeout=15)
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
            "homeRuns_pitching": stats_pitching.get('homeRuns', 0),
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

    return [TeamStats(**row) for row in rows]

def fetch_player_stats(game_pk: str, max_retries: int) -> list[PlayerStats]:
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    resp = retry_request(url, max_retries=max_retries)
    #resp = requests.get(url, timeout=15)
    resp.raise_for_status() # checks if http request was successful
    data = resp.json()
    teams = ['home', 'away']
    rows = []

    for side in teams:
        team_players = data['teams'][side]['players']

        for player_id, player_data in team_players.items():
            
            stats_batting = player_data.get('stats', {}).get('batting', {})
            stats_pitching = player_data.get('stats', {}).get('pitching', {})
            stats_fielding = player_data.get('stats', {}).get('fielding', {})
            
            row = {
                "game_pk": game_pk,
                "team_side": side,
                "player_id": player_data.get('person',{}).get('id'),
                "player_name": player_data.get('person',{}).get('fullName'),

                # Batting
                "at_bats": stats_batting.get('atBats'),
                "runs_scored": stats_batting.get('runs'),
                "hits": stats_batting.get('hits'),
                "home_runs": stats_batting.get('homeRuns'),
                "rbis": stats_batting.get('rbi'),
                "walks_batting": stats_batting.get('baseOnBalls'),
                "strikeouts_batting": stats_batting.get('strikeOuts'),
                "left_on_base": stats_batting.get('leftOnBase'),
                "stolen_bases": stats_batting.get('stolenBases'),

                # Pitching
                "innings_pitched": stats_pitching.get('inningsPitched'),
                "hits_allowed": stats_pitching.get('hits'),
                "runs_allowed": stats_pitching.get('runs'),
                "earned_runs": stats_pitching.get('earnedRuns'),
                "strikeouts_pitching": stats_pitching.get('strikeOuts'),
                "walks_pitching": stats_pitching.get('baseOnBalls'),
                "pitches_thrown": stats_pitching.get('numberOfPitches'),

                #Fielding
                "putouts": stats_fielding.get('putOuts'),
                "assists": stats_fielding.get('assists'),
                "errors": stats_fielding.get('errors'),
            }

            rows.append(row)

    return [PlayerStats(**row) for row in rows]



if __name__ == "__main__":
    #print(fetch_games(date.today().isoformat())[:2])  # quick sanity check
    #print("\n\n")
    #print(fetch_team_stats(634627, max_retries=3))
    #print("\n")
    #print(fetch_games_for_date(date.today().isoformat(), max_retries=3))
    print(fetch_player_stats(634627, max_retries=3))

