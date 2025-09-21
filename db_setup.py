import duckdb
import pandas as pd
from config import DUCKDB_PATH


def load_csvs_to_duckdb():
    con = duckdb.connect(DUCKDB_PATH)
    print("shaari")
    # Load CSVs into DuckDB
    df_measurements = pd.read_csv("data/measurements.csv")
    df_cycles = pd.read_csv("data/cycles.csv")
    df_floats = pd.read_csv("data/floats.csv")
    
    con.execute("CREATE OR REPLACE TABLE measurements AS SELECT * FROM df_measurements")
    print("shankar")
    con.execute("CREATE OR REPLACE TABLE cycles AS SELECT * FROM df_cycles")
    con.execute("CREATE OR REPLACE TABLE floats AS SELECT * FROM df_floats")
    
    print("✅ CSVs loaded into DuckDB successfully!")
    return con

def query_db(con, sql):
    df = con.execute(sql).df()
    return df

if __name__ == "__main__":
    con = load_csvs_to_duckdb()
    
    # Optional: test query
    df_test = query_db(con, "SELECT COUNT(*) AS total_measurements FROM measurements")
    print(df_test)
    print("✅ Database setup complete!")