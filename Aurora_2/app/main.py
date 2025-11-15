from __future__ import annotations

import os
import re
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from groq import Groq


# ---- Upstream API ----
# ask the API for up to 1000 messages in one page instead of the default 100
API_URL = "http://november7-730026606190.europe-west1.run.app/messages/?limit=1000"
CACHE_TTL_SEC = 300  # seconds


# ---- App ----
app = FastAPI(title="Aurora QA Service (LLM + Retrieval)")


class Answer(BaseModel):
    answer: str


# ---- Globals / cache ----
_cache: Dict[str, Any] = {"data": None, "fetched_at": 0.0}

# LLM client
GROQ_MODEL = "llama-3.3-70b-versatile"
_groq_client: Optional[Groq] = None


def _init_llm_client() -> Groq:
    """Lazy-init the Groq client, fail clearly if no API key."""
    global _groq_client
    if _groq_client is not None:
        return _groq_client

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")

    _groq_client = Groq(api_key=api_key)
    return _groq_client


# ---- Utils ----

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _message_text(m: Any) -> str:
    """
    Turn a raw message dict into text we can index/search.
    We include the user_name so questions with the member's name work.
    """
    if isinstance(m, dict):
        base = m.get("text") or m.get("content") or m.get("message") or ""
        user_name = m.get("user_name") or m.get("member", {}).get("name") if isinstance(m.get("member"), dict) else m.get("user_name")
        if user_name:
            # e.g. "Fatima El-Tahir: I'd like a table for four..."
            base = f"{user_name}: {base}"
        return _normalize(base)
    return _normalize(str(m))


async def _fetch_messages() -> List[Any]:
    """
    Fetch and normalize the upstream response to a list of messages.
    Cached for CACHE_TTL_SEC seconds.
    """
    now = time.time()
    if _cache["data"] is not None and now - _cache["fetched_at"] < CACHE_TTL_SEC:
        return _cache["data"]

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.get(API_URL)
        r.raise_for_status()
        data = r.json()

    messages: Any = None

    if isinstance(data, dict):
        # Try common wrapper keys
        for key in ("messages", "items", "data", "results"):
            if key in data and isinstance(data[key], list):
                messages = data[key]
                break

        # Defensive: one level deeper
        if messages is None:
            for v in data.values():
                if isinstance(v, dict):
                    for key in ("messages", "items", "data", "results"):
                        if key in v and isinstance(v[key], list):
                            messages = v[key]
                            break
                if messages is not None:
                    break
    else:
        messages = data

    if not isinstance(messages, list):
        raise ValueError(f"Unexpected API response shape: {type(data)}")

    _cache["data"] = messages
    _cache["fetched_at"] = now
    return messages


def _top_k_messages(question: str, messages: List[Any], k: int = 20) -> List[str]:
    """
    Return the texts of the k most relevant messages for this question,
    using a simple TF-IDF + cosine similarity ranking.
    """
    texts = [_message_text(m) for m in messages]
    texts = [t for t in texts if t]

    if not texts:
        return []

    vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words="english")
    mat = vec.fit_transform(texts)
    qv = vec.transform([question])
    sims = linear_kernel(qv, mat).ravel()
    idxs = sims.argsort()[::-1][:k]

    return [texts[i] for i in idxs]


def _extractive_fallback(question: str, messages: List[Any], top_k: int = 5) -> str:
    """
    Old-school fallback: directly return a relevant sentence/snippet without using the LLM.
    Used if the LLM fails or no key is set.
    """
    texts = [_message_text(m) for m in messages]
    texts = [t for t in texts if t]

    if not texts:
        return "I couldn’t find relevant messages."

    vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words="english")
    mat = vec.fit_transform(texts)
    qv = vec.transform([question])
    sims = linear_kernel(qv, mat).ravel()
    idxs = sims.argsort()[::-1][:top_k]

    # pick the first sentence from the most relevant text that shares some words
    qwords = set(re.findall(r"[a-z]{3,}", question.lower()))
    for i in idxs:
        sents = re.split(r"(?<=[.!?])\s+", texts[i])
        for s in sents:
            s = s.strip()
            if not s:
                continue
            if any(w in s.lower() for w in qwords):
                return s[:300]

    return texts[idxs[0]][:300]


def _llm_answer(question: str, messages: List[Any]) -> str:
    """
    Use Groq LLM (Llama 3.3 70B) to answer the question based only on the messages.
    """
    top_texts = _top_k_messages(question, messages, k=20)
    if not top_texts:
        return "I couldn’t find any messages to base an answer on."

    context = "\n\n---\n\n".join(top_texts[:20])

    user_prompt = f"""
You are a QA assistant for member messages.

You will be given a set of messages from different members, and then a question.
Answer the question ONLY using information in the messages. If the answer is not
present in the messages, say clearly that you cannot find it in the messages.

MESSAGES:
{context}

QUESTION:
{question}

Answer in one short, direct sentence.
""".strip()

    try:
        client = _init_llm_client()
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        )
        content = completion.choices[0].message.content
        return (content or "").strip() or "I couldn't produce an answer."
    except Exception as e:
        # If Groq fails (no key, network, etc.), fall back to extractive QA.
        print("GROQ ERROR:", repr(e))
        return _extractive_fallback(question, messages)


# ---- Startup ----

@app.on_event("startup")
async def warmup():
    """
    Try to fetch messages on startup so first request is faster.
    If this fails, it will be retried on first request.
    """
    try:
        await _fetch_messages()
    except Exception as e:
        print("Warmup skipped:", e)


# ---- Routes ----

@app.get("/ask", response_model=Answer)
async def ask(q: str = Query(..., description="Natural-language question")):
    question = (q or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question.")

    try:
        msgs = await _fetch_messages()
    except Exception as e:
        print("FETCH ERROR:", repr(e))
        # If we can't even get messages, there is nothing we can do.
        return Answer(answer="Upstream message service is unavailable; I can't answer right now.")

    # Always use LLM+retrieval in this approach
    ans = _llm_answer(question, msgs)
    return Answer(answer=ans)


@app.get("/debug/top")
async def debug_top(q: str, k: int = 5):
    """
    Debug endpoint: return the top-k message texts most similar to the query,
    according to TF-IDF. Helpful for understanding what context the LLM sees.
    """
    question = (q or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question.")

    msgs = await _fetch_messages()
    top_texts = _top_k_messages(question, msgs, k=k)
    return {"top": top_texts}


@app.get("/healthz")
def health():
    return {"status": "ok"}
