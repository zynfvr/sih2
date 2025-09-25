# llm_chat.py
import os
import duckdb
import re
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from build_vectordb import build_vector_db
from db_setup import load_csvs_to_duckdb, query_db
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate
from config import GEN_API_KEY, DUCKDB_PATH, VECTOR_DB_PATH


# --- 1️⃣ Configure Gemini ---
if not GEN_API_KEY:
    raise ValueError("❌ No Gemini API Key found in config.py")
genai.configure(api_key=GEN_API_KEY)


# --- 2️⃣ DuckDB Connection ---
def get_db_connection():
    return duckdb.connect(DUCKDB_PATH)


def initialize_database():
    if not os.path.exists(DUCKDB_PATH):
        print("🔄 DuckDB database not found. Loading CSVs...")
        return load_csvs_to_duckdb()
    else:
        print(f"✅ DuckDB database found at {DUCKDB_PATH}")
        return get_db_connection()


con_init = initialize_database()
con_init.close()


# --- 3️⃣ Embedding Model (Retriever brain 🧠) ---
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    cache_folder="./hf_models",
    model_kwargs={"local_files_only": True}
)


# --- 4️⃣ Build or Load Vector DB ---
if not os.path.exists(VECTOR_DB_PATH):
    print("🔄 Vector DB not found. Building vector DB...")
    vectordb = build_vector_db(embedding_model=embedding_model)
else:
    print(f"✅ Loading existing Vector DB from {VECTOR_DB_PATH}")
    vectordb = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_model)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})


# --- 5️⃣ Memory Store ---
MEMORY_DB_PATH = "memory_chroma"
memory_store = Chroma(persist_directory=MEMORY_DB_PATH, embedding_function=embedding_model)


# --- 6️⃣ Session Context ---
session_context = {
    "current_float_id": None,
    "current_region": None,
    "current_parameter": None,
    "recent_queries": []
}


def extract_entities(text):
    """Detect float ID, region, parameter from user text."""
    entities = {}

    float_match = re.search(r"\b(\d{7})\b", text)
    if float_match:
        entities["float_id"] = float_match.group(1)

    region_keywords = ["arctic", "pacific", "atlantic", "indian", "southern", "mediterranean", "arabian"]
    for r in region_keywords:
        if r in text.lower():
            entities["region"] = r
            break

    param_keywords = ["temperature", "salinity", "pressure", "depth", "oxygen", "chlorophyll", "ph"]
    for p in param_keywords:
        if p in text.lower():
            entities["parameter"] = p
            break

    return entities


def query_db_facts(question, entities):
    con = get_db_connection()
    facts = []
    try:
        # Total floats
        res = con.execute("SELECT COUNT(DISTINCT platform_number) FROM floats").fetchone()
        facts.append(f"Database contains {res[0]} unique floats.")

        # Float ID check
        if "float_id" in entities:
            fid = entities["float_id"]
            exists = con.execute("SELECT COUNT(*) FROM floats WHERE platform_number = ?", [fid]).fetchone()[0]
            if exists > 0:
                facts.append(f"Float {fid} exists in the database.")
                # ✅ Now pull latest cycle info
                loc = con.execute("""
                    SELECT cycle_number, latitude, longitude, date
                    FROM cycles
                    WHERE platform_number = ?
                    ORDER BY cycle_number DESC LIMIT 1
                """, [fid]).fetchone()
                if loc:
                    facts.append(f"Last cycle {loc[0]} at {loc[1]:.2f}°N, {loc[2]:.2f}°E on {loc[3]}.")
                else:
                    facts.append(f"No cycle records found for float {fid}.")
            else:
                facts.append(f"Float {fid} not found in database.")

        # Region check...
        if entities.get("region") == "arabian":
            res = con.execute("""
                SELECT DISTINCT f.platform_number 
                FROM floats f JOIN cycles c ON f.platform_number=c.platform_number
                WHERE c.latitude BETWEEN 5 AND 25 
                AND c.longitude BETWEEN 45 AND 78
                LIMIT 5
            """).fetchall()
            if res:
                ids = [str(r[0]) for r in res]
                facts.append(f"Sample floats in Arabian Sea: {', '.join(ids)}.")

    except Exception as e:
        # ⚠️ Don't hide errors, surface them for debugging
        facts.append(f"DATABASE ERROR: {repr(e)}")
        print(f"\n[DB ERROR] {e}\n")
    finally:
        con.close()

    return "\n".join(facts)


def update_context(question):
    """Update active session context."""
    global session_context
    ents = extract_entities(question)
    for k, v in ents.items():
        session_context[f"current_{k}"] = v
    session_context["recent_queries"].append(question)
    if len(session_context["recent_queries"]) > 5:
        session_context["recent_queries"].pop(0)


# --- 7️⃣ System Prompt ---
system_template = """
You are an expert oceanographer and data analyst specialized in Argo float data.
Use BOTH:
1. DuckDB database facts (ground truth numbers, locations, stats)
2. Retrieved documents (semantic context)

Guidelines:
- Always prioritize database facts over text from retrieval.
- Answer in natural, plain language (no raw SQL or raw tables unless explicitly asked).
- If information is missing in DB, clearly say "I don’t know."
- Maintain continuity using the ACTIVE CONTEXT (float ID, region, parameter).
- For vague pronouns ("it", "this float"), resolve them from ACTIVE CONTEXT.
- For plots, generate clean matplotlib code with labels and titles.
- Never hallucinate values not present in the database.
"""

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{context}\n\nQuestion: {question}")]
)


# --- 8️⃣ Ask LLM (Gemini as reasoning + voice 🗣️) ---
def ask_llm(question, db_facts, docs):
    doc_context = "\n".join([d.page_content for d in docs]) if docs else "No retrieved docs."
    full_context = f"=== DATABASE FACTS ===\n{db_facts}\n\n=== RETRIEVED DOCS ===\n{doc_context}"
    prompt_text = prompt_template.format(context=full_context, question=question)

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt_text)
    return response.text.strip()


# --- 9️⃣ Hybrid Answer (DB + RAG + LLM) ---
def hybrid_answer(question):
    ents = extract_entities(question)
    db_facts = query_db_facts(question, ents)

    # Vector + memory retrieval
    docs = []
    try:
        docs += retriever.get_relevant_documents(question)
    except:
        pass
    try:
        docs += memory_store.similarity_search(question, k=2)
    except:
        pass

    # Get final LLM answer
    answer = ask_llm(question, db_facts, docs)

    # Update session
    update_context(question)
    try:
        memory_store.add_texts([answer], metadatas=[{"question": question}])
    except:
        pass

    return answer


# --- 🔟 CLI ---
if __name__ == "__main__":
    print("✅ Argo Float LLM Chat Ready!")
    while True:
        q = input("\n❓ Ask your question (type 'exit' to quit): ")
        if q.lower() in ["exit", "quit"]:
            break
        ans = hybrid_answer(q)
        print("\n💡 Response:\n", ans)

'''
import os
import duckdb
import re
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from build_vectordb import build_vector_db
from db_setup import load_csvs_to_duckdb, query_db
import google.generativeai as genai
from langchain_core.prompts import ChatPromptTemplate
from config import GEN_API_KEY, DUCKDB_PATH, VECTOR_DB_PATH

# --- 1️⃣ Configure Gemini (heavy tasks only) ---
if not GEN_API_KEY:
    raise ValueError("❌ No Gemini API Key found in config.py")
genai.configure(api_key=GEN_API_KEY)

# --- 2️⃣ Database connection function (to avoid file locking) ---
def get_db_connection():
    """Get a new database connection"""
    return duckdb.connect(DUCKDB_PATH)

# --- 3️⃣ Initialize database if needed ---
def initialize_database():
    """Initialize database if it doesn't exist"""
    if not os.path.exists(DUCKDB_PATH):
        print("🔄 DuckDB database not found. Loading CSVs...")
        return load_csvs_to_duckdb()
    else:
        print(f"✅ DuckDB database found at {DUCKDB_PATH}")
        return get_db_connection()

# Initialize database
con_init = initialize_database()
con_init.close()  # Close immediately to avoid file locking

# --- 4️⃣ Setup Embeddings (cached, offline) ---
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    cache_folder="./hf_models",
    model_kwargs={"local_files_only": True}
)

# --- 5️⃣ Build or Load Vector DB ---
if not os.path.exists(VECTOR_DB_PATH):
    print("🔄 Vector DB not found. Building vector DB...")
    vectordb = build_vector_db(embedding_model=embedding_model)  # pass cached embeddings
else:
    print(f"✅ Loading existing Vector DB from {VECTOR_DB_PATH}")
    vectordb = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_model)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# --- 6️⃣ Conversation Memory ---
MEMORY_DB_PATH = "memory_chroma"
if not os.path.exists(MEMORY_DB_PATH):
    memory_store = Chroma(persist_directory=MEMORY_DB_PATH, embedding_function=embedding_model)
else:
    memory_store = Chroma(persist_directory=MEMORY_DB_PATH, embedding_function=embedding_model)

# --- 🆕 Session Context Storage (In-Memory) ---
session_context = {
    "current_float_id": None,
    "current_region": None,
    "current_parameter": None,
    "current_topic": None,
    "recent_queries": [],
    "conversation_summary": ""
}

def extract_context_entities(text):
    """Extract key entities from text to maintain context"""
    entities = {}
    
    # Extract float IDs (various patterns)
    float_patterns = [
        r'float\s+(\d{7})',
        r'platform\s+(\d{7})',
        r'ID\s+(\d{7})',
        r'\b(\d{7})\b'
    ]
    
    for pattern in float_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            entities['float_id'] = matches[0]
            break
    
    # Extract regions
    region_keywords = ['arctic', 'pacific', 'atlantic', 'indian', 'southern', 'mediterranean', 'north', 'south', 'east', 'west']
    for keyword in region_keywords:
        if keyword in text.lower():
            entities['region'] = keyword
            break
    
    # Extract parameters
    param_keywords = ['temperature', 'salinity', 'pressure', 'depth', 'oxygen', 'chlorophyll', 'ph']
    for param in param_keywords:
        if param in text.lower():
            entities['parameter'] = param
            break
    
    return entities

def update_session_context(question, answer):
    """Update session context based on question and answer"""
    global session_context
    
    # Extract entities from both question and answer
    q_entities = extract_context_entities(question)
    a_entities = extract_context_entities(answer)
    
    # Update context with new entities
    for key, value in {**q_entities, **a_entities}.items():
        if key == 'float_id':
            session_context['current_float_id'] = value
        elif key == 'region':
            session_context['current_region'] = value
        elif key == 'parameter':
            session_context['current_parameter'] = value
    
    # Keep recent queries (last 5)
    session_context['recent_queries'].append(question)
    if len(session_context['recent_queries']) > 5:
        session_context['recent_queries'].pop(0)
    
    # Update conversation summary
    if len(session_context['recent_queries']) > 1:
        topics = []
        if session_context['current_float_id']:
            topics.append(f"discussing float {session_context['current_float_id']}")
        if session_context['current_parameter']:
            topics.append(f"analyzing {session_context['current_parameter']}")
        if session_context['current_region']:
            topics.append(f"in {session_context['current_region']} region")
        
        session_context['conversation_summary'] = "Currently " + ", ".join(topics) if topics else ""

def build_context_prompt(question):
    """Build enhanced context for the LLM"""
    context_parts = []
    
    # Add active context
    if session_context['current_float_id']:
        context_parts.append(f"ACTIVE CONTEXT: Currently discussing float {session_context['current_float_id']}")
    
    if session_context['current_parameter']:
        context_parts.append(f"FOCUS PARAMETER: {session_context['current_parameter']}")
    
    if session_context['current_region']:
        context_parts.append(f"REGION OF INTEREST: {session_context['current_region']}")
    
    # Add recent conversation flow
    if len(session_context['recent_queries']) > 1:
        context_parts.append("RECENT CONVERSATION:")
        for i, q in enumerate(session_context['recent_queries'][-3:], 1):  # Last 3 queries
            context_parts.append(f"  {i}. {q}")
    
    # Add conversation summary
    if session_context['conversation_summary']:
        context_parts.append(f"SUMMARY: {session_context['conversation_summary']}")
    
    return "\n".join(context_parts)

# --- 7️⃣ Enhanced System Prompt ---
system_template = """
You are an expert oceanographer and data analyst specialized in Argo float data.
You have access to:
1. DuckDB tables: floats, cycles, measurements
2. Retrieved documents from a vector database
3. Conversation memory (past Q&A)
4. ACTIVE SESSION CONTEXT (very important!)

CRITICAL INSTRUCTIONS FOR CONTEXT CONTINUITY:
- If there is an ACTIVE CONTEXT mentioning a specific float ID, ALWAYS continue discussing that same float unless the user explicitly asks about a different one
- If the user asks vague questions like "show me more", "what about temperature", "tell me about this float", refer to the ACTIVE CONTEXT to understand what they mean
- If discussing a specific region or parameter, maintain that focus unless explicitly changed
- When the user uses pronouns (it, this, that) or vague references, use the ACTIVE CONTEXT to resolve what they're referring to

Guidelines:
- Answer naturally in plain language; do not label the source (SQL/RAG) to the user
- Unless specifically asked, do not show SQL queries or raw data tables
- You can query DuckDB tables for exact numbers, counts, max/min, totals, or patterns but always return your answer in natural plain language
- You can also use retrieved documents and memory to support reasoning
- Combine all available sources as needed to form accurate answers
- Never invent data; clearly note if something is unknown and express it to them
- Provide concise reasoning and oceanographic context if asked in depth
- For visualizations, generate reproducible Python code (matplotlib preferred) with proper titles, labels, and legends
- Be friendly and professional and helpful like a human expert
- MOST IMPORTANTLY: Always check and use the ACTIVE CONTEXT to maintain conversation continuity
"""

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{context}\n\nQuestion: {question}")]
)

# --- 8️⃣ LLM Utility ---
def ask_llm(question: str, docs):
    # Build enhanced context
    session_context_str = build_context_prompt(question)
    
    # Combine all context
    doc_context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No extra context."
    full_context = f"{session_context_str}\n\nRETRIEVED DOCUMENTS:\n{doc_context}"
    
    prompt_text = prompt_template.format(context=full_context, question=question)

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt_text)
    return response.text.strip()

# --- 9️⃣ SQL Execution ---
def execute_sql_from_llm(sql: str):
    try:
        con = get_db_connection()
        df = query_db(con, sql)
        con.close()
        return df
    except Exception as e:
        return f"Error executing SQL: {e}"

# --- 🔟 Enhanced Hybrid Answer ---
def hybrid_answer(question: str):
    """
    Enhanced version that maintains session context
    """
    
    # Retrieve memory
    try:
        memory_docs = memory_store.similarity_search(question, k=3)
    except Exception as e:
        print(f"Warning: Memory search failed: {e}")
        memory_docs = []

    # Retrieve vector docs
    try:
        retrieved_docs = retriever.get_relevant_documents(question)
    except Exception as e:
        print(f"Warning: RAG retrieval failed: {e}")
        retrieved_docs = []

    # Combine all context
    context_docs = memory_docs + retrieved_docs

    # Ask LLM with enhanced context
    answer = ask_llm(question, context_docs)

    # Update session context after getting the answer
    update_session_context(question, answer)

    # Save to memory
    try:
        memory_store.add_texts([answer], metadatas=[{"question": question}])
    except Exception as e:
        print(f"Warning: Failed to save to memory: {e}")

    return answer

# Add function to clear context if needed
def clear_session_context():
    """Clear the session context"""
    global session_context
    session_context = {
        "current_float_id": None,
        "current_region": None,
        "current_parameter": None,
        "current_topic": None,
        "recent_queries": [],
        "conversation_summary": ""
    }

# --- 11️⃣ CLI Loop ---
if __name__ == "__main__":
    print("✅ Argo Float LLM Chat Ready!")

    while True:
        user_question = input("\n❓ Ask your question (type 'exit' to quit, 'clear' to reset context): ")
        if user_question.lower() in ["exit", "quit"]:
            break
        elif user_question.lower() == "clear":
            clear_session_context()
            print("🧹 Session context cleared!")
            continue

        answer = hybrid_answer(user_question)
        print("\n💡 Response:\n", answer)
        
        # Show current context for debugging (remove in production)
        if session_context['current_float_id'] or session_context['current_parameter'] or session_context['current_region']:
            print(f"\n🔍 Current Context: Float={session_context['current_float_id']}, Parameter={session_context['current_parameter']}, Region={session_context['current_region']}")

'''