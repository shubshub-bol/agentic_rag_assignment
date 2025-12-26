import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from graph import app

def main():
    print("====================================================")
    print("      AWS RAG Assistant (CLI)                       ")
    print("====================================================")
    
    # Check if index exists
    if not os.path.exists("faiss_index"):
        print("Error: 'faiss_index' not found.")
        print("Please run 'python ingest.py' first.")
        return

    while True:
        try:
            query = input("\n[User] Enter your question (or 'q' to quit): ").strip()
            if query.lower() in ["q", "quit", "exit"]:
                print("Goodbye!")
                break
            
            if not query:
                continue

            print(f"\n[System] Processing: {query}")
            
            # Initial State
            initial_state = {
                "query": query,
                "query_type": "",
                "plan_description": "",
                "retrieved_chunks": [],
                "final_answer": ""
            }

            # Run Workflow
            # Use invoke for simple synchronous execution
            result = app.invoke(initial_state)

            print("\n====================================================")
            print("FINAL ANSWER")
            print("====================================================")
            print(result.get("final_answer"))
            print("====================================================")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    main()
