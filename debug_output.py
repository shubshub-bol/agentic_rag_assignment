import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def debug_response():
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0, google_api_key=api_key)
    
    # Simple invoke
    try:
        response = llm.invoke("Say 'Hello' only.")
        print(f"Type of content: {type(response.content)}")
        print(f"Content: {response.content}")
        
    except Exception as e:
        print(e)

if __name__ == "__main__":
    debug_response()
