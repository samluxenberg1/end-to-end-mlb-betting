import psycopg2
import logging

# Configure logging for better debugging and traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_sql_file(filename, connection):
    """
    Load and execute SQL script with DB connection.
    """
    with open(filename, "r") as f:
        sql = f.read()
    with connection.cursor() as cursor:
        cursor.execute(sql)
    

def main():

    try:
        with psycopg2.connect(
            host="localhost",
            dbname="mlb_db",
            user="mlb_user",
            password="mlb_pass",
            port=5432
        ) as conn:
            
            run_sql_file(filename="db/transform.sql", connection=conn)
            logging.info("✅ transform.sql executed successfully")
    
    except psycopg2.Error as e:
        logging.info(f"❌ Database error: {e}")
    except FileNotFoundError as e:
        logging.info(f"❌ SQL file not found: {e}")
    except Exception as e:
        logging.info(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
