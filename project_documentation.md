# Local‑First AI Study Assistant – Project Documentation

## Abstract

University students often struggle to sift through large volumes of personal study notes when revising for exams or completing assignments.  Traditional keyword search tools lack contextual understanding and can return irrelevant results, while cloud‑hosted AI systems raise privacy concerns.  This project presents a **Local‑First AI Study Assistant**—a simple, transparent web application that lets users upload plain‑text or Markdown notes, ask questions in natural language, and retrieve the most relevant passages from those notes.  Optionally, the system can use an external language model (LLM) to generate an answer strictly grounded in the uploaded content.  The assistant stores all data locally, runs entirely on the user’s machine, and employs classical information‑retrieval techniques (BM25) alongside modern API integration.  This documentation explains the motivation, design, implementation, and use of the system.

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Stakeholder Analysis](#stakeholder-analysis)
4. [Requirements](#requirements)
5. [Computational Thinking Approach](#computational-thinking-approach)
6. [System Design](#system-design)
    - 6.1 [Architecture](#architecture)
    - 6.2 [Data Model](#data-model)
    - 6.3 [Data Flow](#data-flow)
7. [Setup & Running](#setup--running)
8. [API Overview](#api-overview)
9. [Usage Guide](#usage-guide)
10. [Agile Development](#agile-development)
11. [Evaluation & Future Work](#evaluation--future-work)
12. [Conclusion](#conclusion)

## Introduction

Modern university courses generate a vast amount of lecture notes, reading guides, and personal study materials.  When exam season arrives, students often find themselves scrolling through lengthy documents, trying to remember where a particular concept was explained.  Many AI‑powered note‑taking tools exist, but they typically upload personal data to external servers and may hallucinate answers that are not actually present in the source material.  A **local‑first** study assistant addresses these concerns by processing and storing data on the user’s own machine and by grounding answers exclusively in the user’s notes.

The **Local‑First AI Study Assistant** is a FastAPI web application written in Python.  Users can create logical “courses,” upload their notes in `.txt` or `.md` format, and ask questions in natural language.  The system normalises and chunks the text, indexes it using the BM25 retrieval algorithm, and ranks the most relevant segments.  If the user provides an OpenAI API key, the assistant will call an LLM to generate a concise answer that cites the retrieved passages.  Otherwise, it simply returns the relevant excerpts.

## Problem Statement

Students need a way to **efficiently retrieve information** from large collections of unstructured notes without sacrificing accuracy or privacy.  Keyword searches return too many irrelevant hits, and generic LLM chatbots often fabricate answers that are not present in the source.  Uploading sensitive academic materials to third‑party services may breach data‑protection policies.  Therefore, we require a tool that:

* Runs locally and stores data on the user’s machine.
* Accepts user‑provided notes and indexes them for search.
* Responds to natural language questions with precise, context‑aware excerpts.
* Optionally integrates with a language model while still grounding responses in the uploaded content.

## Stakeholder Analysis

* **Student users:** need quick, reliable access to information in their own notes.  They are concerned about privacy and accuracy.
* **Academic advisors:** are interested in maintaining academic integrity and transparency.  They want assurance that any AI answers cite original sources and do not encourage plagiarism.
* **Developers/maintainers:** require a modular, maintainable codebase that can evolve over time.
* **Institutional IT:** must ensure data privacy, security, and ethical use of AI technologies.

Throughout development, these stakeholders were consulted to identify pain points and desirable features.  The resulting requirements guided the system’s design and priorities.

## Requirements

From the stakeholder discussions, the following functional and non‑functional requirements were established:

1. **Local storage:** All documents, indexes, and embeddings (if used) must be stored locally using a lightweight database (SQLite).
2. **Upload notes:** Users should be able to upload `.txt` or `.md` files associated with a “course.”
3. **Natural language search:** Users can ask questions, and the system should return the most relevant note segments ranked by BM25.
4. **Citation & transparency:** When an LLM answer is generated, it must be based solely on retrieved passages and should cite the source segment IDs.
5. **Offline functionality:** The retrieval and ranking system must work without network access.  LLM integration should be optional.
6. **Simplicity and extensibility:** The API should be easy to use, and the code should be modular enough to support future enhancements (e.g., other file types, vector search).

## Computational Thinking Approach

The project applied several computational‑thinking principles:

* **Decomposition:** The problem was broken down into independent components—file storage, text processing, retrieval, answer generation, and user interface.  Each part was implemented and tested separately.
* **Abstraction:** Real‑world concepts such as courses, documents, and text fragments were represented as database tables and Pydantic models.  This abstraction simplified the logic for storing and retrieving data.
* **Logical & procedural thinking:** A clear pipeline was defined for processing user input: create a course → upload notes → normalise and chunk the text → index with BM25 → accept a query → retrieve the top‑k chunks → (optionally) send them to an LLM.
* **Pattern recognition:** The BM25 algorithm captures term frequencies and inverse document frequencies to rank chunks by relevance to the query.

## System Design

### Architecture

The assistant’s architecture follows a clean separation of concerns:

```
┌───────────────┐    HTTP/JSON    ┌───────────────┐
│   User (UI)   ├────────────────►│ FastAPI Server│
└──────┬────────┘                 └───────────────┘
       │                                  │
       ▼                                  ▼
┌───────────────┐   DB/queries   ┌─────────────────────┐
│   Storage     │◄──────────────►│ Retrieval & Ranking │
│ (SQLite)      │                └─────────────────────┘
└───────────────┘                        │
                                         ▼
                    (optional) ┌─────────────────────┐
                               │    LLM API (OpenAI) │
                               └─────────────────────┘
```

* **User Interface:** A simple HTML/JavaScript page served by FastAPI allows the user to create courses, upload files, and ask questions.
* **FastAPI Server:** Handles HTTP requests, orchestrates calls to the retrieval engine, and returns JSON or HTML responses.
* **Storage Layer:** A local SQLite database stores courses, full documents, individual text chunks, and pre‑computed BM25 statistics.  The schema is defined in `study_assistant/app.py`.
* **Retrieval & Ranking:** Implements text normalisation, chunking (default 1 200 characters per chunk with 180 character overlap), BM25 scoring, and ranking of retrieved chunks.
* **LLM API (optional):** If an `OPENAI_API_KEY` is provided, the system packages the top‑k retrieved chunks into a prompt and calls the OpenAI API.  The response is parsed to ensure it cites the chunk IDs, preventing hallucination.  If no key is provided, the assistant returns only the retrieved passages.

### Data Model

The SQLite database stores several tables:

| Table           | Description                                                    |
|-----------------|----------------------------------------------------------------|
| `courses`       | Course records with a unique name and creation timestamp.      |
| `documents`     | Uploaded files associated with a course (full text).           |
| `chunks`        | Individual text chunks with a reference back to the document.  |
| `bm25_df`       | Term document‑frequency counts per course (for BM25 IDF).      |
| `bm25_stats`    | Cached statistics (total chunks, average length) per course.    |

### Data Flow

The following diagram summarises how data moves through the system when a user asks a question:

```
User Input (question)           ┌─────────────────────┐
  │                             │     Tokenisation    │
  ▼                             └──────────┬──────────┘
Normalized query terms                   │
  │                                      │ BM25 scoring
  ▼                                      ▼
BM25 ranker returns top‑k chunks   ┌───────────────┐
  │                                │   Top‑k       │
  ▼                                │ retrieved     │
Optional: send chunks to LLM      │   chunks      │
  │                                └──────┬────────┘
  ▼                                       │
LLM answer (with citations) or             ▼
raw retrieved chunks          JSON response to UI

**Default `top_k` value:** The number of chunks returned by the BM25 ranker can be controlled via the `top_k` parameter in the `/ask` endpoint.  If omitted, the system uses a sensible default (`DEFAULT_TOP_K = 6` in the source code).  Users may increase or decrease this value to retrieve more or fewer passages for a query.
```

The ranking and (optional) answering steps are deterministic given the same input and notes.  The system never queries external sources other than the LLM API when enabled.

## Setup & Running

1. **Clone and navigate to the project**:

   ```bash
   git clone https://github.com/ajama06/study-assistant.git
   cd study-assistant
   ```

2. **Create and activate a Python virtual environment** (highly recommended):

   On Unix/macOS:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows (cmd.exe):

   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install fastapi uvicorn pydantic python-multipart requests
   ```

4. **Run the application**:

   ```bash
   uvicorn study_assistant.app:app --reload
   ```

   By default, FastAPI will start at `http://127.0.0.1:8000` and serve the web interface.  The SQLite database (`study_assistant.sqlite3`) will be created in the project directory on first run.

5. **Enable the LLM (optional)**:

   To generate AI‑assisted answers, set these environment variables before starting the server:

   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   # Optionally specify a model—default is gpt-4.1-mini
   export OPENAI_MODEL="gpt-4.1-mini"
   export OPENAI_BASE_URL="https://api.openai.com/v1"  # change only if using a custom endpoint
   ```

## API Overview

The backend exposes a small set of REST endpoints; all responses are JSON unless stated otherwise.

| Method & Path                    | Description                                                    |
|----------------------------------|----------------------------------------------------------------|
| `POST /courses`                  | Create a new course.  Body: `{ "name": "string" }`.  Returns the created course with ID. |
| `GET /courses`                   | List all existing courses.                                     |
| `POST /courses/{id}/upload`      | Upload a `.txt` or `.md` file to a course.  Use `multipart/form-data` with fields `file` and `filename`.  Returns the number of chunks created. |
| `POST /courses/{id}/ask`         | Ask a question.  Body: `{ "question": "string", "top_k": integer }`.  The `top_k` field (optional) controls how many chunks to retrieve; if omitted, the system defaults to 6.  Returns the question, the answer (if LLM enabled), and the retrieved chunks with scores and previews. |

## Usage Guide

1. **Create a course:** Navigate to the root URL and enter a course name.  Each course isolates its own documents and index.
2. **Upload notes:** Select a `.txt` or `.md` file and upload it to the chosen course.  The file is normalised, chunked, and indexed.
3. **Ask questions:** Enter a natural language query.  The system ranks your notes using BM25 and returns the most relevant segments.  If you provided an API key, a concise answer is generated from the retrieved text and the LLM’s output is returned with citations.
4. **Iterate:** Upload more notes or create additional courses as needed.  Repeat the process for different subjects or modules.

The web interface is deliberately minimal; advanced users can call the API endpoints directly using `curl` or any HTTP client.

## Agile Development

Development followed an Agile methodology with short iterative sprints.  Core functionality (database schema, BM25 retrieval, file upload) was implemented first, followed by optional LLM integration and the user interface.  After each iteration, the feature was tested manually and feedback from peers was incorporated.  This incremental approach minimised risk and allowed requirements to evolve based on stakeholder feedback.

## Evaluation & Future Work

### Evaluation

The assistant satisfies the original requirements:

* **Local storage & privacy:** All data is stored in a local SQLite database.  No notes are transmitted unless the user opts into LLM integration.
* **Accurate retrieval:** BM25 ranking returns relevant chunks for a wide range of queries.  The ranking is deterministic and transparent.
* **Grounded answers:** When AI is enabled, the LLM is instructed to use only the retrieved passages and to cite their IDs.  This prevents hallucination and maintains academic integrity.
* **Ease of use:** The simple web UI requires no command‑line interaction, and API endpoints are well‑documented.

### Future Enhancements

Possible extensions include:

* **Additional file formats:** Support for PDF, slide decks, or HTML notes via text extraction.
* **Vector search:** Integration of dense embeddings (e.g., via sentence transformers) for semantic search, possibly stored in a vector database.
* **Offline LLM:** Use of an on‑device language model to eliminate external API calls entirely.
* **Authentication:** Add user accounts and authentication for multi‑user scenarios.
* **Analytics:** Track study sessions, highlight frequently asked questions, and visualise progress.

These enhancements would build upon the modular architecture already in place.

## Conclusion

The Local‑First AI Study Assistant combines classical information retrieval (BM25) with optional modern language model integration to create a tool that is both **effective** and **trustworthy**.  By storing data locally and grounding answers in user‑provided content, it respects privacy and academic integrity.  The system illustrates how computational thinking and Agile development can be applied to solve a real‑world educational problem, providing a strong foundation for future innovation.
