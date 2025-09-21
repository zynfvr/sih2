from llm_chat import create_rag_pipeline, ask_question

def main():
    con, qa_chain = create_rag_pipeline()
    
    print("✅ System ready! Ask questions about Argo float data.")
    while True:
        user_question = input("\n❓ Question: ")
        if user_question.lower() in ["exit", "quit"]:
            break
        answer = ask_question(user_question, con, qa_chain)
        print("\n💡 Answer:\n", answer)

if __name__ == "__main__":
    main()
