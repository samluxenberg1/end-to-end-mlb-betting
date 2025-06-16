import logging
import psycopg2

DB_CONFIG = {
    "dbname": "mlb_db",
    "user": "mlb_user", 
    "password": "mlb_pass",
    "host": "localhost",
    "port": 5432,
    }

# Configure logging for better debugging and traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_sql_file(filename, connection):
    """
    Load and execute SQL script with DB connection.
    Handles multiple SQL queries separated by semicolons.
    """
    with open(filename, "r") as f:
        sql_content = f.read()
    
    # Split on semicolons and execute each statement
    queries = [query.strip() for query in sql_content.split(';') if query.strip()]
    
    with connection.cursor() as cursor:
        for i, query in enumerate(queries, 1):
            if query:
                logging.info(f"Executing query {i}/{len(queries)}")
                logging.info(f"Query is: {query}")
                try:
                    cursor.execute(query)
                    if cursor.description:
                        rows = cursor.fetchall()
                        column_names = [desc[0] for desc in cursor.description]
                        logging.info(f"Results: {len(rows)} rows with values: {rows[0][0]}")

                        # Print column headers
                        print("\n" + " | ".join(column_names))
                        print("-" * (len(" | ".join(column_names))))

                        # Print rows
                        for row in rows:
                            print(" | ".join(str(val) for val in row))
                        print()

                except Exception as e:
                    logging.error(f"Error executing query {i}: {e}")
                    logging.error(f"Query was: {query}")
    

def main():

    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            
            run_sql_file(filename="db/data_validation_checks.sql", connection=conn)
            logging.info("✅ data_validation_checks.sql executed successfully")
    
    except psycopg2.Error as e:
        logging.error(f"❌ Database error: {e}")
    except FileNotFoundError as e:
        logging.error(f"❌ SQL file not found: {e}")
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
