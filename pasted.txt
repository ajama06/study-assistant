# file: study_assistant/app.py
"""
Local-First AI Study Assistant (RAG) — BM25 retrieval + optional LLM answering.

Run:
  python -m venv .venv && source .venv/bin/activate
  pip install fastapi uvicorn pydantic python-multipart requests

Start:
  uvicorn study_assistant.app:app --reload

Open:
  http://127.0.0.1:8000

Optional LLM:
  export OPENAI_API_KEY="..."
  export OPENAI_MODEL="gpt-4.1-mini"   # or any model you have access to
"""
from __future__ import annotations

import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "study_assistant.sqlite3"

# ----------------------------
# Config
# ----------------------------
K1 = 1.5
B = 0.75
DEFAULT_TOP_K = 6

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()


# ----------------------------
# Utilities
# ----------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9_']+")


def now_ts() -> int:
    return int(time.time())


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 180) -> List[str]:
    """
    Chunk by paragraph blocks; if blocks are huge, hard-split.
    """
    text = normalize_text(text)
    if not text:
        return []

    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        merged = "\n\n".join(buf).strip()
        if merged:
            chunks.append(merged)
        buf = []
        buf_len = 0

    for block in blocks:
        if len(block) > max_chars:
            flush()
            start = 0
            while start < len(block):
                end = min(len(block), start + max_chars)
                part = block[start:end].strip()
                if part:
                    chunks.append(part)
                start = max(0, end - overlap)
            continue

        if buf_len + len(block) + 2 <= max_chars:
            buf.append(block)
            buf_len += len(block) + 2
        else:
            flush()
            buf.append(block)
            buf_len = len(block) + 2

    flush()

    # Light de-dup (can happen with overlap slicing)
    deduped: List[str] = []
    seen = set()
    for c in chunks:
        key = hash(c[:300])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


# ----------------------------
# DB Layer
# ----------------------------
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS courses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  course_id INTEGER NOT NULL,
  filename TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  FOREIGN KEY(course_id) REFERENCES courses(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id INTEGER NOT NULL,
  chunk_index INTEGER NOT NULL,
  text TEXT NOT NULL,
  token_count INTEGER NOT NULL,
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Cached DF per course for BM25 IDF
CREATE TABLE IF NOT EXISTS bm25_df (
  course_id INTEGER NOT NULL,
  term TEXT NOT NULL,
  df INTEGER NOT NULL,
  PRIMARY KEY(course_id, term),
  FOREIGN KEY(course_id) REFERENCES courses(id) ON DELETE CASCADE
);

-- Cached stats per course
CREATE TABLE IF NOT EXISTS bm25_stats (
  course_id INTEGER PRIMARY KEY,
  chunk_count INTEGER NOT NULL,
  avg_chunk_len REAL NOT NULL,
  updated_at INTEGER NOT NULL,
  FOREIGN KEY(course_id) REFERENCES courses(id) ON DELETE CASCADE
);
"""


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def db_init() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with db_connect() as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def db_course_exists(conn: sqlite3.Connection, course_id: int) -> bool:
    row = conn.execute("SELECT 1 FROM courses WHERE id = ?", (course_id,)).fetchone()
    return row is not None


def db_get_course_id_by_name(conn: sqlite3.Connection, name: str) -> Optional[int]:
    row = conn.execute("SELECT id FROM courses WHERE name = ?", (name,)).fetchone()
    return int(row["id"]) if row else None


def db_create_course(conn: sqlite3.Connection, name: str) -> int:
    ts = now_ts()
    cur = conn.execute("INSERT INTO courses(name, created_at) VALUES(?,?)", (name, ts))
    conn.commit()
    return int(cur.lastrowid)


def db_list_courses(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, name, created_at FROM courses ORDER BY created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def db_insert_document_and_chunks(
    conn: sqlite3.Connection, course_id: int, filename: str, content: str, chunks: List[str]
) -> int:
    ts = now_ts()
    cur = conn.execute(
        "INSERT INTO documents(course_id, filename, content, created_at) VALUES(?,?,?,?)",
        (course_id, filename, content, ts),
    )
    doc_id = int(cur.lastrowid)

    for i, ch in enumerate(chunks):
        tc = len(tokenize(ch))
        conn.execute(
            "INSERT INTO chunks(document_id, chunk_index, text, token_count) VALUES(?,?,?,?)",
            (doc_id, i, ch, tc),
        )

    conn.commit()
    return doc_id


def db_get_course_chunks(conn: sqlite3.Connection, course_id: int) -> List[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT
          c.id AS chunk_id,
          c.text AS text,
          c.token_count AS token_count,
          d.id AS document_id,
          d.filename AS filename
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.course_id = ?
        """,
        (course_id,),
    ).fetchall()
    return rows


def db_rebuild_bm25_cache(conn: sqlite3.Connection, course_id: int) -> None:
    """
    Recomputes DF(term) and avg chunk length for the course.
    """
    chunks = db_get_course_chunks(conn, course_id)

    term_df: Dict[str, int] = {}
    total_len = 0
    for r in chunks:
        tokens = set(tokenize(r["text"]))
        total_len += int(r["token_count"])
        for t in tokens:
            term_df[t] = term_df.get(t, 0) + 1

    conn.execute("DELETE FROM bm25_df WHERE course_id = ?", (course_id,))
    conn.execute("DELETE FROM bm25_stats WHERE course_id = ?", (course_id,))

    for term, df in term_df.items():
        conn.execute(
            "INSERT INTO bm25_df(course_id, term, df) VALUES(?,?,?)",
            (course_id, term, df),
        )

    chunk_count = len(chunks)
    avg_len = (total_len / chunk_count) if chunk_count else 0.0
    conn.execute(
        "INSERT INTO bm25_stats(course_id, chunk_count, avg_chunk_len, updated_at) VALUES(?,?,?,?)",
        (course_id, chunk_count, float(avg_len), now_ts()),
    )
    conn.commit()


def db_get_bm25_stats(conn: sqlite3.Connection, course_id: int) -> Tuple[int, float]:
    row = conn.execute(
        "SELECT chunk_count, avg_chunk_len FROM bm25_stats WHERE course_id = ?",
        (course_id,),
    ).fetchone()
    if not row:
        return (0, 0.0)
    return (int(row["chunk_count"]), float(row["avg_chunk_len"]))


def db_get_df_map(conn: sqlite3.Connection, course_id: int, terms: Sequence[str]) -> Dict[str, int]:
    if not terms:
        return {}
    placeholders = ",".join("?" for _ in terms)
    rows = conn.execute(
        f"SELECT term, df FROM bm25_df WHERE course_id = ? AND term IN ({placeholders})",
        (course_id, *terms),
    ).fetchall()
    return {str(r["term"]): int(r["df"]) for r in rows}


# ----------------------------
# BM25 Retrieval
# ----------------------------
@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_id: int
    filename: str
    score: float
    text: str


def bm25_idf(N: int, df: int) -> float:
    # Classic BM25+ style smoothing
    return math.log(1.0 + (N - df + 0.5) / (df + 0.5))


def bm25_score(
    *,
    query_terms: List[str],
    chunk_terms: List[str],
    chunk_len: int,
    avg_len: float,
    N: int,
    df_map: Dict[str, int],
) -> float:
    if not query_terms or not chunk_terms or N <= 0 or avg_len <= 0:
        return 0.0

    tf: Dict[str, int] = {}
    for t in chunk_terms:
        tf[t] = tf.get(t, 0) + 1

    score = 0.0
    for term in query_terms:
        df = df_map.get(term, 0)
        if df <= 0:
            continue
        f = tf.get(term, 0)
        if f <= 0:
            continue

        idf = bm25_idf(N, df)
        denom = f + K1 * (1 - B + B * (chunk_len / avg_len))
        score += idf * (f * (K1 + 1)) / (denom + 1e-9)

    return score


def retrieve_top_k(conn: sqlite3.Connection, course_id: int, question: str, top_k: int) -> List[RetrievedChunk]:
    question = normalize_text(question)
    q_terms = tokenize(question)
    if not q_terms:
        return []

    N, avg_len = db_get_bm25_stats(conn, course_id)
    if N <= 0:
        return []

    df_map = db_get_df_map(conn, course_id, list(set(q_terms)))
    rows = db_get_course_chunks(conn, course_id)

    scored: List[RetrievedChunk] = []
    for r in rows:
        text = str(r["text"])
        chunk_terms = tokenize(text)
        s = bm25_score(
            query_terms=q_terms,
            chunk_terms=chunk_terms,
            chunk_len=int(r["token_count"]),
            avg_len=avg_len,
            N=N,
            df_map=df_map,
        )
        if s > 0:
            scored.append(
                RetrievedChunk(
                    chunk_id=int(r["chunk_id"]),
                    document_id=int(r["document_id"]),
                    filename=str(r["filename"]),
                    score=float(s),
                    text=text,
                )
            )

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[: max(1, top_k)]


# ----------------------------
# Optional OpenAI Answering
# ----------------------------
class LLMAnswer(BaseModel):
    answer: str
    cited_chunk_ids: List[int] = Field(default_factory=list)


def openai_chat_completion(prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a careful study assistant. Only use provided sources."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=45)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI error {r.status_code}: {r.text[:500]}")
    data = r.json()
    return data["choices"][0]["message"]["content"]


def build_prompt(question: str, retrieved: List[RetrievedChunk]) -> str:
    sources = []
    for i, ch in enumerate(retrieved, start=1):
        sources.append(
            f"[Source {i} | chunk_id={ch.chunk_id} | file={ch.filename}]\n{ch.text}\n"
        )
    sources_txt = "\n".join(sources) if sources else "(no sources)"
    return (
        "Answer the question using ONLY the sources below.\n"
        "If the sources are insufficient, say what is missing.\n"
        "Cite sources by chunk_id in parentheses, e.g. (chunk_id=12).\n\n"
        f"Question:\n{question}\n\n"
        f"Sources:\n{sources_txt}\n"
    )


def fallback_answer(question: str, retrieved: List[RetrievedChunk]) -> LLMAnswer:
    if not retrieved:
        return LLMAnswer(
            answer="I couldn’t find anything in your notes that matches that question yet. Upload more notes or rephrase.",
            cited_chunk_ids=[],
        )
    stitched = "\n\n".join(
        f"- From {ch.filename} (chunk_id={ch.chunk_id}):\n  {ch.text}"
        for ch in retrieved
    )
    return LLMAnswer(
        answer=(
            "I don’t have an LLM configured, so here are the most relevant excerpts from your notes.\n\n"
            f"{stitched}\n\n"
            "If you set OPENAI_API_KEY, I can generate a grounded answer with citations."
        ),
        cited_chunk_ids=[ch.chunk_id for ch in retrieved],
    )


def answer_question(question: str, retrieved: List[RetrievedChunk]) -> LLMAnswer:
    if not OPENAI_API_KEY:
        return fallback_answer(question, retrieved)

    prompt = build_prompt(question, retrieved)
    text = openai_chat_completion(prompt)

    cited = sorted({int(x) for x in re.findall(r"chunk_id\s*=\s*(\d+)", text)})
    return LLMAnswer(answer=text.strip(), cited_chunk_ids=cited)


# ----------------------------
# API Models
# ----------------------------
class CreateCourseRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=80)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)


# ----------------------------
# FastAPI App
# ----------------------------
db_init()
app = FastAPI(title="Local-First AI Study Assistant (RAG)", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Study Assistant (RAG)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; max-width: 980px; }
    .row { display: flex; gap: 18px; flex-wrap: wrap; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; flex: 1; min-width: 320px; }
    input, textarea, select, button { width: 100%; padding: 10px; margin-top: 8px; border-radius: 10px; border: 1px solid #ccc; }
    button { cursor: pointer; }
    pre { white-space: pre-wrap; background: #f7f7f7; padding: 12px; border-radius: 10px; overflow-x: auto; }
    small { color: #666; }
  </style>
</head>
<body>
  <h1>Study Assistant (RAG)</h1>
  <p><small>Upload notes per course, then ask questions. Optional: set <code>OPENAI_API_KEY</code> for grounded answers.</small></p>

  <div class="row">
    <div class="card">
      <h2>1) Create / Select Course</h2>
      <input id="courseName" placeholder="Course name (e.g., CS101 Algorithms)" />
      <button onclick="createCourse()">Create course</button>
      <div style="margin-top:12px">
        <select id="courseSelect"></select>
        <button onclick="refreshCourses()">Refresh courses</button>
      </div>
      <pre id="courseOut"></pre>
    </div>

    <div class="card">
      <h2>2) Upload Notes</h2>
      <input id="file" type="file" accept=".txt,.md" />
      <button onclick="upload()">Upload to selected course</button>
      <pre id="uploadOut"></pre>
    </div>

    <div class="card">
      <h2>3) Ask</h2>
      <textarea id="question" rows="5" placeholder="Ask something from your notes..."></textarea>
      <input id="topK" type="number" min="1" max="20" value="6" />
      <button onclick="ask()">Ask</button>
      <pre id="askOut"></pre>
    </div>
  </div>

<script>
async function refreshCourses() {
  const r = await fetch('/courses');
  const data = await r.json();
  const sel = document.getElementById('courseSelect');
  sel.innerHTML = '';
  for (const c of data.courses) {
    const opt = document.createElement('option');
    opt.value = c.id;
    opt.textContent = `${c.name} (id=${c.id})`;
    sel.appendChild(opt);
  }
  document.getElementById('courseOut').textContent = JSON.stringify(data, null, 2);
}

async function createCourse() {
  const name = document.getElementById('courseName').value.trim();
  const r = await fetch('/courses', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name})
  });
  const data = await r.json();
  document.getElementById('courseOut').textContent = JSON.stringify(data, null, 2);
  await refreshCourses();
}

async function upload() {
  const courseId = document.getElementById('courseSelect').value;
  const f = document.getElementById('file').files[0];
  if (!courseId) return alert('Select a course');
  if (!f) return alert('Choose a .txt or .md file');

  const form = new FormData();
  form.append('file', f);

  const r = await fetch(`/courses/${courseId}/upload`, { method: 'POST', body: form });
  const data = await r.json();
  document.getElementById('uploadOut').textContent = JSON.stringify(data, null, 2);
}

async function ask() {
  const courseId = document.getElementById('courseSelect').value;
  const question = document.getElementById('question').value.trim();
  const top_k = parseInt(document.getElementById('topK').value, 10) || 6;
  if (!courseId) return alert('Select a course');
  if (!question) return alert('Type a question');

  const r = await fetch(`/courses/${courseId}/ask`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question, top_k})
  });
  const data = await r.json();
  document.getElementById('askOut').textContent = JSON.stringify(data, null, 2);
}

refreshCourses();
</script>
</body>
</html>
"""


@app.get("/courses")
def list_courses() -> JSONResponse:
    with db_connect() as conn:
        return JSONResponse({"courses": db_list_courses(conn)})


@app.post("/courses")
def create_course(req: CreateCourseRequest) -> JSONResponse:
    name = req.name.strip()
    with db_connect() as conn:
        existing = db_get_course_id_by_name(conn, name)
        if existing is not None:
            return JSONResponse({"course_id": existing, "name": name, "created": False})
        cid = db_create_course(conn, name)
        return JSONResponse({"course_id": cid, "name": name, "created": True})


@app.post("/courses/{course_id}/upload")
async def upload_notes(course_id: int, file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = (Path(file.filename).suffix or "").lower()
    if ext not in {".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Only .txt or .md supported in this MVP")

    raw = await file.read()
    try:
        content = raw.decode("utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode upload: {e}")

    content = normalize_text(content)
    chunks = chunk_text(content)

    if not chunks:
        raise HTTPException(status_code=400, detail="No usable text found after normalization/chunking")

    with db_connect() as conn:
        if not db_course_exists(conn, course_id):
            raise HTTPException(status_code=404, detail="Course not found")

        doc_id = db_insert_document_and_chunks(conn, course_id, file.filename, content, chunks)
        db_rebuild_bm25_cache(conn, course_id)

    return JSONResponse(
        {
            "course_id": course_id,
            "document_id": doc_id,
            "filename": file.filename,
            "chunks_created": len(chunks),
        }
    )


@app.post("/courses/{course_id}/ask")
def ask(course_id: int, req: AskRequest) -> JSONResponse:
    question = req.question.strip()
    top_k = int(req.top_k)

    with db_connect() as conn:
        if not db_course_exists(conn, course_id):
            raise HTTPException(status_code=404, detail="Course not found")

        retrieved = retrieve_top_k(conn, course_id, question, top_k=top_k)
        answer = answer_question(question, retrieved)

    citations = [
        {
            "chunk_id": ch.chunk_id,
            "document_id": ch.document_id,
            "filename": ch.filename,
            "score": round(ch.score, 4),
            "text_preview": (ch.text[:400] + "…") if len(ch.text) > 400 else ch.text,
        }
        for ch in retrieved
    ]

    return JSONResponse(
        {
            "question": question,
            "top_k": top_k,
            "answer": answer.model_dump(),
            "retrieved": citations,
            "llm_enabled": bool(OPENAI_API_KEY),
        }
    )


# file: study_assistant/__init__.py
# (intentionally empty; keeps this as a package)
