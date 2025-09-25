import duckdb

# Connect to your DB
con = duckdb.connect("argo.duckdb")

# List all tables
tables = con.execute("SHOW TABLES").fetchall()
print("Tables:", tables)

# Inspect schema of a table
schema = con.execute("DESCRIBE cycles").fetchall()
print("Schema of cycles:", schema)

# Preview data
df = con.execute("SELECT * FROM cycles LIMIT 5").df()
print(df)
