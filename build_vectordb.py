# build_vectordb.py
import os
import duckdb
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import DUCKDB_PATH, VECTOR_DB_PATH

# --- Build Vector DB ---
def build_vector_db():
    # Connect to DuckDB
    con = duckdb.connect(DUCKDB_PATH)

    # Load all cycle-level info
    df_cycles = con.execute("""
        SELECT FLOAT_ID, PROFILE_NUMBER, CYCLE_NUMBER, JULD, LATITUDE, LONGITUDE,
               POSITION_QC, DIRECTION, DATA_MODE,
               PROFILE_PRES_QC, PROFILE_TEMP_QC, PROFILE_PSAL_QC
        FROM cycles
    """).df()

    # Build contextual texts for embeddings
    texts = df_cycles.apply(
        lambda row: (
            f"Float {row['FLOAT_ID']} | Profile {row['PROFILE_NUMBER']} | "
            f"Cycle {row['CYCLE_NUMBER']} | Date: {row['JULD']} | "
            f"Location: ({row['LATITUDE']}, {row['LONGITUDE']}) | "
            f"Position QC: {row['POSITION_QC']} | Direction: {row['DIRECTION']} | "
            f"Data Mode: {row['DATA_MODE']} | "
            f"Pressure QC: {row['PROFILE_PRES_QC']} | "
            f"Temperature QC: {row['PROFILE_TEMP_QC']} | "
            f"Salinity QC: {row['PROFILE_PSAL_QC']}"
        ),
        axis=1
    ).tolist()

    # Metadata for filtering / retrieval
    metadatas = df_cycles.to_dict(orient="records")

    # --- Initialize FREE HuggingFace embeddings ---
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"  # world-class free model
    )

    # Build Chroma vector DB
    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=VECTOR_DB_PATH
    )

    

    vectordb.persist()
    print(f"✅ Vector DB created with HuggingFace embeddings at {VECTOR_DB_PATH}")
    return vectordb

# --- Optional main block for testing ---
if __name__ == "__main__":
    build_vector_db()
