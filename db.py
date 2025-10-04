import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

NEON_CONN = os.environ.get("NEON_CONN")

def get_conn():
    return psycopg2.connect(NEON_CONN, connect_timeout=10)

def upload_csv(run_name: str, table_name: str, csv_path: str):
    """Upload CSV to NeonDB. If connection fails, log error but don't crash."""
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return
        
        full_table = f"run_{run_name}_{table_name}"
        
        with get_conn() as conn:
            with conn.cursor() as cur:
                cols = ", ".join([f"{c} TEXT" for c in df.columns])
                cur.execute(f"CREATE TABLE IF NOT EXISTS {full_table} ({cols});")
                
                values = df.values.tolist()
                execute_values(
                    cur,
                    f"INSERT INTO {full_table} ({', '.join(df.columns)}) VALUES %s",
                    values
                )
            conn.commit()
        print(f"[DB] Successfully uploaded {table_name} to NeonDB")
        
    except psycopg2.OperationalError as e:
        print(f"[DB WARNING] Connection to NeonDB failed: {e}")
        print(f"[DB WARNING] Continuing training without uploading {table_name}...")
    except Exception as e:
        print(f"[DB ERROR] Unexpected error uploading {table_name}: {e}")
        print(f"[DB ERROR] Continuing training...")