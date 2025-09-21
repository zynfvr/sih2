import os
import getpass
import duckdb
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai
from build_vectordb import build_vector_db
from db_setup import query_db, load_csvs_to_duckdb
from config import GEN_API_KEY, VECTOR_DB_PATH


# --- 1️⃣ Configure Gemini ---
if not GEN_API_KEY:
    GEN_API_KEY = getpass.getpass("Enter your Google Gemini API Key: ")

genai.configure(api_key=GEN_API_KEY)

# --- 2️⃣ Connect DuckDB ---
con = load_csvs_to_duckdb()

# --- 3️⃣ Build / Connect Vector DB ---
vectordb = build_vector_db()
retriever = vectordb.as_retriever()

# --- 4️⃣ Create a prompt template for Argo queries ---
system_template = (
    "You are an expert oceanographer. "
    "You have access to DuckDB tables containing Argo float data. "
    "Use SQL queries only if needed. "
    "Answer questions clearly in natural language."
)

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{question}")]
)

# --- 5️⃣ Ask LLM function ---
def ask_llm(question: str):
    # Format prompt
    prompt = prompt_template.invoke({"question": question})
    
    # Convert prompt to messages
    messages = prompt.to_messages()
    
    # Call Gemini
    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(messages)
    
    return response.text

# --- 6️⃣ Optional: execute example SQL if LLM mentions it ---
def execute_sql_from_llm(sql: str):
    try:
        df = query_db(con, sql)
        return df
    except Exception as e:
        return f"Error executing SQL: {e}"

# --- 7️⃣ CLI testing ---
if __name__ == "__main__":
    print("✅ Argo Float LLM Chat Ready!")
    
    while True:
        user_question = input("\n❓ Ask your question (type 'exit' to quit): ")
        if user_question.lower() in ["exit", "quit"]:
            break
        
        # Step 1: Retrieve relevant cycle info from vector DB (optional)
        # For demo, we just get top 3 cycles
        docs = retriever.get_relevant_documents(user_question)
        top_cycles = [doc.metadata['cycle_number'] for doc in docs[:3]]
        print(f"🔹 Top relevant cycles from vector DB: {top_cycles}")
        
        # Step 2: Ask LLM
        llm_answer = ask_llm(user_question)
        print("\n💡 LLM Response:\n", llm_answer)
        
        # Optional: LLM may suggest SQL execution
        # sql_example = "SELECT * FROM measurements WHERE cycle_id=10"
        # df_result = execute_sql_from_llm(sql_example)
        # print("\n📊 SQL Result:\n", df_result)
