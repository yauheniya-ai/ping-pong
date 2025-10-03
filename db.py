import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# NeonDB connection string (from dashboard or environment)
NEON_CONN = os.environ.get("NEON_CONN")

def get_conn():
    return psycopg2.connect(NEON_CONN)

def upload_csv(run_name: str, table_name: str, csv_path: str):
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    # prepend run_name to table for uniqueness
    full_table = f"run_{run_name}_{table_name}"

    # create table if not exists
    with get_conn() as conn:
        with conn.cursor() as cur:
            # generate columns dynamically
            cols = ", ".join([f"{c} TEXT" for c in df.columns])
            cur.execute(f"CREATE TABLE IF NOT EXISTS {full_table} ({cols});")

            # insert rows
            values = df.values.tolist()
            execute_values(
                cur,
                f"INSERT INTO {full_table} ({', '.join(df.columns)}) VALUES %s",
                values
            )
            conn.commit()
