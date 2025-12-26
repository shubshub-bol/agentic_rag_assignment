import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class SynthesizerAgent:
    def __init__(self):
        # Using Google Gemini. Expects GOOGLE_API_KEY in env.
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Warning: GOOGLE_API_KEY not found in environment during Init.")
        self.llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0, google_api_key=api_key)

    def synthesize(self, state: dict) -> dict:
        """
        Synthesizes the final answer using retrieved context.
        """
        print(f"\n--- [Synthesizer] Generating answer... ---")
        
        query = state.get("query")
        chunks = state.get("retrieved_chunks", [])
        
        if not chunks:
            return {"final_answer": "I could not find any relevant information in the AWS RAG guide."}

        # Format context
        context_text = ""
        for i, chunk in enumerate(chunks):
            context_text += f"--- Chunk {i+1} ---\n{chunk['content']}\n"

        # System Prompt
        system_prompt = (
            "You are a helpful assistant for the AWS RAG options guide. "
            "Use ONLY the provided context to answer the user's question. "
            "If the answer is not in the context, say 'This information is not available in the provided AWS RAG guide.' "
            "Do not use outside knowledge. "
            "Cite your sources loosely by referring to the context if possible (though the user just wants the answer). "
            "Keep the answer simple, structured, and readable."
        )

        user_prompt = f"Question: {query}\n\nContext:\n{context_text}"

        messages = [
            ("system", system_prompt),
            ("user", user_prompt),
        ]

        # Invoke LLM
        response = self.llm.invoke(messages)
        
        # Parse Content (Gemini can return list of parts if grounding is involved)
        content = response.content
        final_text = ""
        
        if isinstance(content, str):
            final_text = content
        elif isinstance(content, list):
            # Extract text from parts
            parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
            final_text = "\n".join(parts)
        else:
            final_text = str(content)

        print("\n[Synthesizer] Answer Generated.")
        return {"final_answer": final_text}
