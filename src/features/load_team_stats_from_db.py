import os
import pandas as pd
import psycopg2
import warnings
import logging

from dotenv import load_dotenv
load_dotenv()

def load_team_stats_from_db():

    DB_CONFIG = {
        "dbname": os.environ['DB_NAME'],
        "user": os.environ['DB_USER'],
        "password": os.environ['DB_PASSWORD'],
        "host": os.environ['DB_HOST'],
        "port": os.environ['DB_PORT']
    }
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Connect to DB and read in data to pandas dataframe
    try: 
        with psycopg2.connect(**DB_CONFIG) as conn:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message = "pandas only supports SQLAlchemy")

                query = """SELECT * FROM team_stats"""
                df_team_stats = pd.read_sql_query(query, conn)

        logging.info(f"Successully loaded {len(df_team_stats)} rows from database")
        
        
        # Preview the data
        print(f"There are {len(df_team_stats)} team-games for  modeling...")
        #print(df_games.head())
    except psycopg2.Error as e: 
        logging.error(f"Database error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

    return df_team_stats



    


