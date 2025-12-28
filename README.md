# Local-First AI Study Assistant

This repository contains a FastAPI web application that lets you upload course notes (`.txt` or `.md`), index them using BM25 retrieval, and optionally answer questions using an OpenAI language model.

## Prerequisites

- Python 3.8 or higher
- A command-line terminal
- (Recommended) A virtual environment for Python packages

## Setup

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/ajama06/study-assistant.git
   cd study-assistant
   ```

2. **Create and activate a virtual environment**.

   **On Unix/macOS**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **On Windows (cmd.exe)**:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn pydantic python-multipart requests
   ```

## Running the application

1. **Initialize and start the server**:
   ```bash
   uvicorn study_assistant.app:app --reload
   ```

   By default, the server will start at `http://127.0.0.1:8000`. The SQLite database (`study_assistant.sqlite3`) will be created automatically on first run.

2. **Open your browser** and go to [http://127.0.0.1:8000](http://127.0.0.1:8000). You can:
   - Create a course
   - Upload your course notes (`.txt` or `.md`) for indexing
   - Ask questions about your notes

## Optional: Enable LLM answering

If you want the app to generate answers using OpenAI's API, set the following environment variables before starting the server:

```bash
export OPENAI_API_KEY="your-openai-api-key"
# Optional: specify a model if you have access to a specific version
export OPENAI_MODEL="gpt-4.1-mini"
```

If these variables are not set, the app will return only the retrieved text excerpts and indicate that the LLM is disabled.

---

For more implementation details, see the source code in `study_assistant/app.py`.
