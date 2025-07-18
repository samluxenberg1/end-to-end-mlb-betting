import csv
import logging
import psycopg2
from psycopg2.extras import execute_values # efficient bulk insert

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_CONFIG = {
    "dbname": "mlb_db",
    "user": "mlb_user", 
    "password": "mlb_pass",
    "host": "localhost",
    "port": 5432,
    }

CSV_PATH_GAMES = "data/raw_games.csv"
CSV_PATH_TEAM_STATS = "data/raw_team_stats.csv"
CSV_PATH_PLAYER_STATS = "data/raw_player_stats.csv"


def clean_row(row):
    """
    Replace empty strings in a dictionary with None (SQL NULL).
    This ensures that numeric fields can be inserted into the database correctly.
    """
    return {key: (None if val == '' else val) for key, val in row.items()}

def load_games_to_db(
        db_config: dict, 
        data: list = None, 
        from_memory: bool = False, 
        csv_path: str = None
        ) -> None:
    """
    Bulk-insert a csv file of games (from fetch_games) into the `games` table.
    Uses ON CONFLICT DO UPDATE so that if a game with 'Scheduled' state (no result)
    is encountered, it will be updated with a new version containing game results.

    Parameters
    ----------
    db_config: dict
        Database connection configuration
    data: list = None
        List of game dictionaries (used when from_memory=True)
    from_memory: bool = False
        If True, load from data parameter, else from CSV file
    csv_path: str = None
        Path to CSV file (required when from_memory=False)
    """
    # Connect to Postgres 
    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cur:
            
            # Determine source of data and prepare rows
            if not from_memory:
                if not csv_path:
                    raise ValueError("csv_path must be provided when from_memory is False")
                # Load from CSV
                with open(csv_path, newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    rows = list(reader)
            else:
                if data is None:
                    raise ValueError("data must be provided when from_memory is True")
                # Load from memory
                #rows = [clean_row(row.model_dump()) for row in data] # no pydantic model for games at the moment...
                rows = [clean_row(row) for row in data]

            if not rows:
                logging.warning("No rows found in data. Exiting.")
                return 
            
            columns = [
                "game_id", 
                "game_date",
                "game_date_time",
                "home_team_id",
                "away_team_id", 
                "home_team", 
                "away_team", 
                "home_score", 
                "away_score", 
                "state", 
                "venue", 
                "game_type"
                ]
            
            # Clean rows if they came from CSV
            if not from_memory:
                rows = [clean_row(row) for row in rows]

            values = [[row[col] for col in columns] for row in rows]

            # Construct the ON CONFLICT DO UPDATE part
            update_columns = ["game_date","game_date_time", "home_score", "away_score", "state", "venue"]
            update_set_clauses = [f"{col} = EXCLUDED.{col}" for col in update_columns]
            update_clause = ", ".join(update_set_clauses)

            sql = f"""
                INSERT INTO games ({', '.join(columns)})
                VALUES %s
                ON CONFLICT (game_id) DO UPDATE SET
                    {update_clause}
                WHERE games.state = 'Scheduled';
            """

            logging.info(f"Inserting {len(values)} rows into the database...")
            
            execute_values(cur, sql, values)
            conn.commit()
            logging.info("Insert complete.")


def load_team_stats_to_db(db_config: dict, data: list = None, from_memory: bool =False, csv_path: str = None) -> None:
    """
    Bulk-insert a csv file of team stats (from fetch_team_stats) into the `team_stats` table.
    Uses ON CONFLICT DO NOTHING so reruns are idempotent. 
    """
    # Connect to Postgres 
    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cur:

            # Determine source of data and prepare rows
            if not from_memory:
                if not csv_path:
                    raise ValueError("csv_path must be provided when from_memory is False")
                
                # Load from CSV
                with open(csv_path, newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    rows = list(reader)
            else:
                if data is None:
                    raise ValueError("data must be provided when from_memory is True")
                # Load from memory
                rows = [clean_row(row.model_dump()) for row in data]

            if not rows:
                logging.warning("No rows found. Exiting.")
                return 
            
            columns = [
                "game_pk",
                "team_side",
                "team_id",
                "runs_batting",
                "hits_batting",
                "strikeOuts_batting",
                "baseOnBalls_batting",
                "avg",
                "obp",
                "slg",
                "pitchesThrown",
                "balls_pitching",
                "strikes_pitching",
                "strikeOuts_pitching",
                "baseOnBalls_pitching",
                "hits_pitching",
                "earnedRuns",
                "homeRuns_pitching",
                "runs_pitching",
                "era",
                "whip",
                "groundOuts_pitching",
                "airOuts_pitching",
                "total",
                "putOuts",
                "assists",
                "errors",
                "doublePlays",
                "triplePlays",
                "rangeFactor",
                "caughtStealing",
                "passedBall",
                "innings"
            ]

            # Clean rows if they came from CSV (memory objects are already cleaned)
            if not from_memory:
                rows = [clean_row(row) for row in rows]
            
            values = [[row[col] for col in columns] for row in rows]

            # Construct the ON CONFLICT DO UPDATE part
            update_columns = list(set(columns)-set(['game_pk','team_side','team_id']))
            update_set_clauses = [f"{col} = EXCLUDED.{col}" for col in update_columns]
            update_clause = ", ".join(update_set_clauses)

            sql = f"""
                INSERT INTO team_stats ({', '.join(columns)})
                VALUES %s
                ON CONFLICT (game_pk, team_side) DO UPDATE SET
                    {update_clause}
                WHERE team_stats.pitchesthrown = 0;
            """

            logging.info(f"Inserting {len(values)} rows into the database...")
            execute_values(cur, sql, values)
            conn.commit()
            logging.info("Insert complete.")

def load_player_stats_to_db(db_config: dict, data: list = None, from_memory: bool = False, csv_path: str = None) -> None:
    """
    Bulk-insert a csv file of player stats (from fetch_player_stats) into the `player_stats` table.
    Uses ON CONFLICT DO NOTHING so reruns are idempotent. 
    """
    # Connect to Postgres 
    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cur:

            # Determine source of data and prepare rows
            if not from_memory:
                if not csv_path:
                    raise ValueError("csv_path must be provided when from_memory is False")

                # Load from CSV
                with open(csv_path, newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    rows = list(reader)
            else:
                if data is None:
                    raise ValueError("data must be provided when from_memory is True")

                # Load from memory
                rows = [clean_row(row.model_dump()) for row in data]

            if not rows:
                logging.warning("No rows found in CSV. Exiting.")
                return 
            
            columns = [

                # Game & Player Metadata
                "game_pk",
                "team_side",
                "team_id",
                "player_id",
                "player_name",

                # Batting Stats
                "at_bats",  
                "runs_scored",
                "hits",
                "home_runs",
                "rbis",
                "walks_batting",
                "strikeouts_batting",
                "left_on_base",
                "stolen_bases",

                # Pitching Stats
                "innings_pitched",
                "hits_allowed",
                "runs_allowed",
                "earned_runs",
                "strikeouts_pitching",
                "walks_pitching",
                "pitches_thrown",

                # Fielding Stats
                "putouts",
                "assists",
                "errors",
            ]

            # Clean rows if they came from CSV (memory objects are already cleaned)
            if not from_memory:
                rows = [clean_row(row) for row in rows]
            
            values = [[row[col] for col in columns] for row in rows]

            sql = f"""
                INSERT INTO player_stats ({', '.join(columns)})
                VALUES %s
                ON CONFLICT (game_pk, player_id) DO NOTHING;
            """

            logging.info(f"Inserting {len(values)} rows into the database...")
            execute_values(cur, sql, values)
            conn.commit()
            logging.info("Insert complete.")
                


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--load-games", action="store_true", help="Load games data into database")
    parser.add_argument("--load-team-stats", action="store_true", help="Load team stats data into database")
    parser.add_argument("--load-player-stats", action="store_true", help="Load player stats data into database")
    args = parser.parse_args()

    if args.load_games:
        load_games_to_db(csv_path=CSV_PATH_GAMES, db_config=DB_CONFIG)
    if args.load_team_stats:
        load_team_stats_to_db(csv_path=CSV_PATH_TEAM_STATS, db_config=DB_CONFIG)
    if args.load_player_stats:
        load_player_stats_to_db(csv_path=CSV_PATH_PLAYER_STATS, db_config=DB_CONFIG)