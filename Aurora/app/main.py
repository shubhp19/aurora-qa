from __future__ import annotations

import re
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import date, timedelta
import zoneinfo

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from dateutil import parser as dateparser

# ---- Upstream API ----
API_URL = "http://november7-730026606190.europe-west1.run.app/messages/"  # note http + trailing slash
CACHE_TTL_SEC = 300

# ---- App ----
app = FastAPI(title="Aurora QA Service")

class Answer(BaseModel):
    answer: str

# ---- Globals / cache ----
_cache = {"data": None, "fetched_at": 0.0}
_vectorizer: Optional[TfidfVectorizer] = None
_matrix = None
_corpus: List[str] = []
_msg_index: List[Dict[str, Any]] = []
_member_names: List[str] = []

# ---- Utils ----
LOCAL_TZ = zoneinfo.ZoneInfo("America/New_York")
WEEKDAY_IDX = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
}

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _message_text(m: Any) -> str:
    """
    Turn a raw message dict into text we can index/search.
    We include the user name so name-based questions work better.
    """
    if isinstance(m, dict):
        base = m.get("text") or m.get("content") or m.get("message") or ""
        user_name = m.get("user_name") or (m.get("member") or {}).get("name") if isinstance(m.get("member"), dict) else m.get("user_name")
        if user_name:
            base = f"{user_name}: {base}"
        return _normalize(base)
    return _normalize(str(m))


async def _fetch_messages() -> List[Any]:
    """Fetch and normalize the upstream response to a list of messages."""
    now = time.time()
    if _cache["data"] is not None and now - _cache["fetched_at"] < CACHE_TTL_SEC:
        return _cache["data"]

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.get(API_URL)
        r.raise_for_status()
        data = r.json()

    messages: Any = None
    if isinstance(data, dict):
        for key in ("messages", "items", "data", "results"):
            if key in data and isinstance(data[key], list):
                messages = data[key]
                break
        if messages is None:
            # defensive: one level deeper
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

def _build_corpus(messages: List[Any]) -> None:
    """Vectorize message texts and remember seen member names (if present)."""
    global _vectorizer, _matrix, _corpus, _msg_index, _member_names
    _msg_index = messages
    _corpus = []
    names = set()

    for msg in messages:
        text = _message_text(msg)
        if isinstance(msg, dict):
            member = msg.get("member") or msg.get("user") or {}
            if isinstance(member, dict):
                n = (member.get("name") or member.get("full_name") or member.get("display") or "")
                if n:
                    names.add(_normalize(n))
                    text = f"{text} {n}"
        _corpus.append(text)

    if not _corpus:
        _vectorizer = None
        _matrix = None
        _member_names = []
        return

    _vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words="english")
    _matrix = _vectorizer.fit_transform(_corpus)
    _member_names = sorted(names, key=str.lower)

# ---- Name/City parsing & sentence helpers ----
def _first_cap_words(s: str) -> list[str]:
    # capture proper-looking tokens, optionally ending with "'s"
    raw = re.findall(r"\b([A-Z][a-z]{2,})(?:'s)?\b", s)
    stop = {"What", "When", "How", "Where", "Which", "Who"}
    # strip any trailing 's from possessive and drop WH-words
    return [w.rstrip("'s") for w in raw if w not in stop]

def _name_and_city_from_question(q: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (person_name, city). Handles:
    - "to <City>", "in <City>"
    - "the <City> trip", "<City> trip"
    - possessives like "Amira's"
    - avoids using the city as the person's name
    """
    city = None
    who: Optional[str] = None

    # --- City detection ---
    m = re.search(r"\bto\s+([A-Z][a-zA-Z]+)\b", q)
    if not m:
        m = re.search(r"\bin\s+([A-Z][a-zA-Z]+)\b", q)
    if not m:
        m = re.search(r"\bthe\s+([A-Z][a-zA-Z]+)\s+trip\b", q)  # "the Dubai trip"
    if not m:
        m = re.search(r"\b([A-Z][a-zA-Z]+)\s+trip\b", q)        # "Dubai trip"
    if m:
        city = m.group(1)

    # --- Person detection ---
    m_poss = re.search(r"\b([A-Z][a-z]+)'s\b", q)  # Amira's
    if m_poss:
        who = m_poss.group(1)
    else:
        m2 = re.search(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", q)  # Vikram Desai
        if m2:
            who = f"{m2.group(1)} {m2.group(2)}"
        else:
            caps = re.findall(r"\b([A-Z][a-z]{2,})\b", q)
            stop = {"What","When","How","Where","Which","Who"}
            caps = [w for w in caps if w not in stop]
            who = caps[0] if caps else None

    # don't accidentally use the city as the person
    if who and city and who.lower() == city.lower():
        who = None

    return who, city

def _filter_by_name_in_text(messages: List[Any], name: Optional[str]) -> List[Any]:
    if not name:
        return messages
    target = name.lower()
    out = []
    for m in messages:
        if target in _message_text(m).lower():
            out.append(m)
    return out or messages

def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def _messages_to_sentences(messages: List[Any]) -> list[str]:
    sents = []
    for m in messages:
        sents.extend(_sentences(_message_text(m)))
    return sents

def _filter_sents(sents: list[str], *needles: str) -> list[str]:
    needs = [n.lower() for n in needles if n]
    out = []
    for s in sents:
        ls = s.lower()
        if all(n in ls for n in needs):
            out.append(s)
    return out

# ---- Relative weekday handling ----
def resolve_relative_weekday(token: str, today: Optional[date] = None) -> Optional[date]:
    """
    Convert 'next friday' / 'this friday' into a concrete date (NY time).
    'this <weekday>': upcoming occurrence this week (including today if matches)
    'next <weekday>': occurrence in the next week
    """
    m = re.search(r"\b(next|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", token, re.I)
    if not m:
        return None
    which = m.group(1).lower()
    wd = WEEKDAY_IDX[m.group(2).lower()]
    today = today or date.today()
    today_idx = today.weekday()

    # distance to this week's occurrence
    delta = (wd - today_idx) % 7
    if which == "this":
        # keep delta as-is (0..6)
        pass
    else:  # "next"
        delta = delta + 7 if delta == 0 else delta + 7
    return today + timedelta(days=delta)

# ---- Extractors ----
def _extract_trip_when(messages: List[Any], person_name: Optional[str], city: Optional[str]) -> Optional[str]:
    sents = _messages_to_sentences(messages)
    city = city or "London"

    travel_needles = ("trip", "travel", "flying", "flight", "going")
    cand: list[str] = []
    for w in travel_needles:
        cand += _filter_sents(sents, person_name, city, w)
    if not cand:
        for w in travel_needles:
            cand += _filter_sents(sents, city, w)
    if not cand:
        return None

    date_like = re.compile(r"\b(?:\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?|\w+\s+\d{1,2}(?:,\s*\d{4})?)\b", re.I)
    rel_like = re.compile(r"\b(?:next|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)

    candidates: list[date] = []

    for s in cand:
        # absolute dates
        for tok in date_like.findall(s):
            try:
                dt = dateparser.parse(tok, fuzzy=True)
                if dt:
                    candidates.append(dt.date())
            except Exception:
                pass
        # relative weekdays (“next Friday”, “this Friday”)
        for m in rel_like.finditer(s):
            full_phrase = m.group(0)  # e.g., "next Friday"
            rd = resolve_relative_weekday(full_phrase)
            if rd:
                candidates.append(rd)

    if candidates:
        candidates.sort()
        return candidates[0].strftime("%B %d, %Y")
    return None

def _extract_how_many_cars(messages: List[Any], person_name: Optional[str]) -> Optional[str]:
    sents = _messages_to_sentences(messages)
    cand = _filter_sents(sents, person_name, "car") + _filter_sents(sents, person_name, "cars")
    if not cand:
        cand = _filter_sents(sents, "cars")

    num_re = re.compile(r"\b(\d+)\b")
    word_to_num = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9}
    word_re = re.compile(r"\b(" + "|".join(word_to_num.keys()) + r")\b", re.I)

    guesses: list[int] = []
    for s in cand:
        guesses += [int(x) for x in num_re.findall(s)]
        for w in word_re.findall(s):
            guesses.append(word_to_num[w.lower()])

    if guesses:
        from collections import Counter
        return str(Counter(guesses).most_common(1)[0][0])
    return None

def _extract_favorite_restaurants(messages: List[Any], person_name: Optional[str]) -> Optional[str]:
    sents = _messages_to_sentences(messages)
    cand = _filter_sents(sents, person_name, "favorite", "restaurant")
    if not cand:
        cand = _filter_sents(sents, "favorite", "restaurant")
    if not cand:
        return None

    proper = re.compile(r"\b([A-Z][A-Za-z0-9&'’\-]+(?:\s+[A-Z][A-Za-z0-9&'’\-]+){0,3})\b")
    noise = {"i","we","they","favorite","restaurants","restaurant","and","the","in","of","for","with","to","at","on"}

    names: list[str] = []
    for s in cand:
        for n in proper.findall(s):
            if n.strip().lower() not in noise and len(n) >= 3:
                names.append(n.strip())
    if names:
        seen = set()
        ordered = [n for n in names if not (n in seen or seen.add(n))]
        return ", ".join(ordered[:5])
    return None

# ---- Fallback ----
def _extractive_answer(question: str, messages: List[Any], top_k: int = 5) -> str:
    texts = [_message_text(m) for m in messages]
    if not texts:
        return "I couldn’t find relevant messages."
    vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words="english")
    mat = vec.fit_transform(texts)
    qv = vec.transform([question])
    sims = linear_kernel(qv, mat).ravel()
    idx = sims.argsort()[::-1][:top_k]
    for i in idx:
        sents = _sentences(texts[i])
        for s in sents:
            if any(w in s.lower() for w in re.findall(r"[a-z]{3,}", question.lower())):
                return s[:300]
    return texts[idx[0]][:300]

#---Answer---
def _best_snippet_for(question: str, texts: list[str]) -> str:
    vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words="english")
    mat = vec.fit_transform(texts)
    qv = vec.transform([question])
    sims = linear_kernel(qv, mat).ravel()
    idx = sims.argsort()[::-1][0]
    # choose the most relevant sentence in that text
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", texts[idx]) if s.strip()]
    if not sents:
        return texts[idx][:300]
    qwords = set(re.findall(r"[a-z]{3,}", question.lower()))
    for s in sents:
        if any(w in s.lower() for w in qwords):
            return s[:300]
    return sents[0][:300]

def _answer_with_source_trip(question: str, messages: list) -> tuple[Optional[str], Optional[str]]:
    who, city = _name_and_city_from_question(question)
    subset = _filter_by_name_in_text(messages, who)
    ans = _extract_trip_when(subset, who, city)
    if ans:
        texts = [_message_text(m) for m in subset]
        return ans, _best_snippet_for(question, texts)
    return None, None

def _answer_with_source_cars(question: str, messages: list) -> tuple[Optional[str], Optional[str]]:
    who, _ = _name_and_city_from_question(question)
    subset = _filter_by_name_in_text(messages, who)
    ans = _extract_how_many_cars(subset, who)
    if ans:
        texts = [_message_text(m) for m in subset]
        return ans, _best_snippet_for(question, texts)
    return None, None

def _answer_with_source_restaurants(question: str, messages: list) -> tuple[Optional[str], Optional[str]]:
    who, _ = _name_and_city_from_question(question)
    subset = _filter_by_name_in_text(messages, who)
    ans = _extract_favorite_restaurants(subset, who)
    if ans:
        texts = [_message_text(m) for m in subset]
        return ans, _best_snippet_for(question, texts)
    return None, None


# ---- Startup ----
@app.on_event("startup")
async def warmup():
    try:
        msgs = await _fetch_messages()
        _build_corpus(msgs)
    except Exception as e:
        print("Warmup skipped:", e)  # lazy-init on first request

# ---- Debug (optional) ----
@app.get("/debug/top")
async def debug_top(q: str, k: int = 5):
    msgs = await _fetch_messages()
    texts = [_message_text(m) for m in msgs]
    if not texts:
        return {"top": []}
    vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words="english")
    mat = vec.fit_transform(texts)
    sims = linear_kernel(vec.transform([q]), mat).ravel()
    idxs = sims.argsort()[::-1][:k]
    return {"top": [texts[i] for i in idxs]}

# ---- Routes ----
@app.get("/ask", response_model=Answer)
async def ask(q: str = Query(..., description="Natural-language question")):
    try:
        question = (q or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Empty question.")

        msgs = await _fetch_messages()
        if not _corpus or len(_msg_index) != len(msgs):
            _build_corpus(msgs)

        who, city = _name_and_city_from_question(question)
        subset = _filter_by_name_in_text(msgs, who)

        qlow = question.lower()

        # Trips
        if (("when" in qlow) or ("what day" in qlow)) and any(w in qlow for w in ("trip","travel","flying","flight","going")):
            ans = _extract_trip_when(subset, who, city)
            if ans:
                return Answer(answer=ans)
            return Answer(answer=f"No messages mention {who or 'that member'} planning a trip to {city or 'that city'}.")

        # Cars
        if ("how many" in qlow) and ("car" in qlow):
            ans = _extract_how_many_cars(subset, who)
            if ans:
                return Answer(answer=ans)
            return Answer(answer=f"No messages mention how many cars {who or 'that member'} has.")

        # Restaurants
        if ("favorite" in qlow) and ("restaurant" in qlow):
            ans = _extract_favorite_restaurants(subset, who)
            if ans:
                return Answer(answer=ans)
            if who:
                return Answer(answer=f"No messages mention {who}'s favorite restaurants.")
            else:
                return Answer(answer="No messages mention that member's favorite restaurants.")


        # Fallback for other questions
        return Answer(answer=_extractive_answer(question, subset))

    except Exception as e:
        print("ASK ERROR:", repr(e))
        try:
            msgs = await _fetch_messages()
        except Exception:
            msgs = []
        return Answer(answer=_extractive_answer(q, msgs))

@app.get("/healthz")
def health():
    return {"status": "ok"}

@app.get("/ask_verbose")
async def ask_verbose(q: str):
    question = (q or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question.")

    msgs = await _fetch_messages()
    if not _corpus or len(_msg_index) != len(msgs):
        _build_corpus(msgs)

    qlow = question.lower()

    if (("when" in qlow) or ("what day" in qlow)) and any(w in qlow for w in ("trip","travel","flying","flight","going")):
        ans, src = _answer_with_source_trip(question, msgs)
        if ans:
            return {"answer": ans, "source_snippet": src}
        who, city = _name_and_city_from_question(question)
        return {"answer": f"No messages mention {who or 'that member'} planning a trip to {city or 'that city'}."}

    if ("how many" in qlow) and ("car" in qlow):
        ans, src = _answer_with_source_cars(question, msgs)
        if ans:
            return {"answer": ans, "source_snippet": src}
        who, _ = _name_and_city_from_question(question)
        return {"answer": f"No messages mention how many cars {who or 'that member'} has."}

    if ("favorite" in qlow) and ("restaurant" in qlow):
        ans, src = _answer_with_source_restaurants(question, msgs)
        if ans:
            return {"answer": ans, "source_snippet": src}
        who, _ = _name_and_city_from_question(question)
        if who:
            return {"answer": f"No messages mention {who}'s favorite restaurants."}
        return {"answer": "No messages mention that member's favorite restaurants."}

    # fallback with provenance
    texts = [_message_text(m) for m in msgs]
    return {"answer": _extractive_answer(question, msgs), "source_snippet": _best_snippet_for(question, texts)}

