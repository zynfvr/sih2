# config.py
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()  # by default, looks for .env in current directory

# Read variables
GEN_API_KEY = os.getenv("GOOGLE_API_KEY")
DUCKDB_PATH = os.getenv("DUCKDB_PATH")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")


