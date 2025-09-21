# build_vectordb.py
import os
import duckdb
import pandas as pd
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from config import DUCKDB_PATH, VECTOR_DB_PATH

# --- Configure Gemini API ---
if not os.environ.get("GOOGLE_API_KEY"):
    import getpass
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# --- Custom wrapper for Gemini embeddings ---
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [genai.embed_text(input=text)["embedding"] for text in texts]

    def embed_query(self, text):
        return genai.embed_text(input=text)["embedding"]

# --- Build Vector DB ---
def build_vector_db():
    con = duckdb.connect(DUCKDB_PATH)

    # Load all cycle-level info
    df_cycles = con.execute("""
        SELECT float_id, PROFILE_NUMBER, CYCLE_NUMBER, JULD, LATITUDE, LONGITUDE,
               POSITION_QC, DIRECTION, DATA_MODE,
               PROFILE_PRES_QC, PROFILE_TEMP_QC, PROFILE_PSAL_QC
        FROM cycles
    """).df()

    # Build richer contextual texts for embeddings
    texts = df_cycles.apply(
        lambda row: (
            f"Float {row['float_id']} | Profile {row['PROFILE_NUMBER']} | "
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

    # Metadata for retrieval filtering
    metadatas = df_cycles.to_dict(orient="records")

    embeddings = GeminiEmbeddings()

    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=VECTOR_DB_PATH
    )

    vectordb.persist()
    print(f"✅ Vector DB created with rich contextual info at {VECTOR_DB_PATH}")
    return vectordb
   
# --- Optional main block for testing ---
if __name__ == "__main__":
    build_vector_db()
