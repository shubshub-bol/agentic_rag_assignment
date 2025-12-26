# Agentic RAG Assistant (Gemini Edition)

A simple, CLI-based Agentic RAG assistant that answers questions specifically from the "Retrieval Augmented Generation options and architectures on AWS" PDF.

Built with Python, LangGraph, Google Gemini, and FAISS. Designed for clarity and learning.

## Features
- **Strict Source of Truth**: Answers only from the provided PDF.
- **Linear Agentic Workflow**: Planner -> Retriever -> Synthesizer.
- **Semantic Search**: Uses `sentence-transformers` and FAISS for vector retrieval.
- **Free LLM**: Powered by Google Gemini (`gemini-flash-latest`).

![Agentic RAG Demo](demo.gif)![demo](https://github.com/user-attachments/assets/f8ffd863-559f-4f62-baf5-96d69a9e08b3)



## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment**:
   Copy `.env.example` to `.env` and add your Google API Key:
   ```bash
   GOOGLE_API_KEY=AIzaSy...
   ```
   > You can get a free key from [Google AI Studio](https://aistudio.google.com/).

3. **PDF Source**:
   The PDF `retrieval-augmented-generation-options.pdf` is **included in this repository**. You do not need to download it manually.

## ⚠️ Important for New Users

If you are cloning this repository, you **MUST** configure the API key for the application to work.

1.  **Create `.env` file**: 
    The `.env` file is ignored by Git (security). You must create it manually in the project root.
2.  **Add Key**:
    File content must look like this:
    ```bash
    GOOGLE_API_KEY=your_actual_api_key_here
    ```
    *(No quotes, no spaces around the equals sign)*

3.  **Install Dependencies**:
    Always run `pip install -r requirements.txt` to ensure you have the correct library versions.

## Usage

### 1. Ingestion
Process the PDF and create the vector database:
```bash
python ingest.py
```
*This creates a local `faiss_index` folder.*

### 2. Run Query
Start the interactive assistant:
```bash
python query.py
```



### 3. Quick Demo (Optional)
Run a script to verify the system without interactive input:
```bash
python demo_rag.py
```

### Example Output
```text
[System] Processing: What is RAG?

[Planner] Analyzing query...
Decision: Query Type = definition

[Retriever] Searching for context...
Found chunk (Score: 1.0): RAG allows you to provide new data...

[Synthesizer] Generating answer...

FINAL ANSWER
====================================================
Retrieval Augmented Generation (RAG) is a technique that...
====================================================
```

## Architecture

**Linear Graph**:
`START` -> `Planner` -> `Retriever` -> `Synthesizer` -> `END`

1. **Planner**: Decides the strategy based on keywords.
2. **Retriever**: Fetches relevant chunks from the FAISS vector store.
3. **Synthesizer**: Uses **Google Gemini** to generate a strict answer from the retrieved chunks.
