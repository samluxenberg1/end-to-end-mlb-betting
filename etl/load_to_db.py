import csv
import logging
from datetime import date
import psycopg2
from psycopg2.extras import execute_values # efficient bulk insert

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_CONFIG = {
    dbname="mlb_db",
    user="mlb_user", 
    password="mlb_pass",
    host="localhost",
    port=5432,
    }

CSV_PATH = "data/raw_games.csv"


def load_games_to_db(csv_path: str, db_config: dict) -> None:
    """
    Bulk-insert a csv file of games (from fetch_games) into the `games` table.
    Uses ON CONFLICT DO NOTHIN so reruns are idempotent. 
    """
    # Connect to Postgres 
    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cur:
            with open(csv_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)

                if not rows:
                    logging.warning("No rows found in CSV. Exiting.")
                    return 
                
                columns = [
                    "game_id", 
                    "game_date", 
                    "home_team", 
                    "away_team", 
                    "home_score", 
                    "away_score", 
                    "state", 
                    "venue", 
                    "game_type"
                    ]
                
                values = [[row[col] for col in columns] for row in rows]

                sql = f"""
                    INSERT INTO games ({', '.join(columns)})
                    VALUES %s
                    ON CONFLICT (game_id) DO NOTHING;
                """

                logging.info(f"Inserting {len(values)} rows into the database...")
                execute_values(cur, sql, values)
                conn.commit()
                logging.info("Insert complete.")
                


if __name__ == "__main__":
    load_games_to_db(CSV_PATH)