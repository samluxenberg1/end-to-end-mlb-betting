from datetime import date
import psycopg2
from psycopg2.extras import execute_values # efficient bulk insert
from extract_games import fetch_games

def get_conn():
    """
    Returns a psycopg2 connection to the local Dockerized Postgres. 
    """
    return psycopg2.connect(
        dbname="mlb_db",
        user="mlb_user",
        password="mlb_pass",
        host="localhost",
        port=5432
    )

def load_games_to_db(rows: list[dict]) -> None:
    """
    Bulk-insert a list of dicts (from fetch_games) into the `games` table. 
    Uses ON CONFLICT DO NOTHING so reruns are idempotent. ???????????
    """
    if not rows:
        print("No games to load.")
        return 
    
    keys = ("game_id", "game_date", "home_team", "away_team", "home_score", "away_score")
    values = [[row[k] for k in keys] for row in rows]

    sql = """
        INSERT INTO games
            (game_id, game_date, home_team, away_team, home_score, away_score)
        VALUES %s
        ON CONFLICT (game_id) DO NOTHING;
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, values)
        conn.commit()

    print(f"Loaded {len(values)} into games")


if __name__ == "__main__":
    today = date.today().isoformat()
    game_rows = fetch_games(today)
    load_games_to_db(game_rows)