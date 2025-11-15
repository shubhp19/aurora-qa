# Aurora QA – Question Answering over Member Messages 

This repo contains **two implementations** of a small question-answering service: one rule-based + TF-IDF approach and one Groq LLM–powered approach, both answering natural-language questions about member data exposed via the November 7 `/messages` API.


Example questions:

- “When is Layla planning her trip to London?”
- “How many cars does Vikram Desai have?”
- “Which restaurant did Fatima El-Tahir request a table at?”
- “What did the member say about the car service?”

The service exposes a single HTTP endpoint:

```http
GET /ask?q=...  →  { "answer": "..." }
```

Internally, it calls the public `/messages` API and either:

- extracts answers using rule-based logic + TF-IDF (classical approach), or  
- retrieves relevant messages and lets a Groq LLM generate the answer (LLM approach).

---

## Repository Layout

```text
aurora-qa/
  Aurora/          # Approach 1: rule-based + TF-IDF (no LLM)
    app/
      main.py      # FastAPI app with handcrafted extractors
      __init__.py
    requirements.txt
    Dockerfile     # optional; not required to run locally

  Aurora_2/        # Approach 2: retrieval + Groq LLM
    app/
      main.py      # FastAPI app that calls Groq LLM
      __init__.py
      DockerFile   # Dockerfile for this app (if used)
    messages.json  # optional local snapshot of /messages for testing
    requirements.txt
```

Both sub-projects expose the same public route: `GET /ask`.

---

## Upstream API

Both services read member messages from the November 7 API:

```http
GET http://november7-730026606190.europe-west1.run.app/messages/
```

The responses are normalized into a list of message objects with fields such as:

```json
{
  "id": "...",
  "user_id": "...",
  "user_name": "Fatima El-Tahir",
  "timestamp": "...",
  "message": "I’d like a table for four at Eleven Madison Park on November 15."
}
```

The QA logic always works strictly from this data; if the messages don’t contain the requested fact, the service answers that the information is not available.

---

## How to Run – Approach 1 (Aurora: TF-IDF + Rules)

This version does **not** use any external LLM. It combines:

- name & entity parsing,
- TF-IDF similarity over message texts,
- task-specific extractors (trip dates, car counts, favorite restaurants),
- and a simple extractive fallback.

### 1. Set up and install

From the repo root:

```bash
cd Aurora

# (optional but recommended)
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the API server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### 3. Example queries

In another terminal:

```bash
curl "http://localhost:8080/ask?q=What%20did%20the%20member%20say%20about%20the%20car%20service%3F"

curl "http://localhost:8080/ask?q=Which%20restaurant%20did%20Fatima%20El-Tahir%20request%20a%20table%20at%3F"

curl "http://localhost:8080/ask?q=When%20is%20Layla%20planning%20her%20trip%20to%20London%3F"
```

This version also exposes a debug endpoint:

- `GET /debug/top?q=...` – show the top-k most similar messages for a query (by TF-IDF).

---

## How to Run – Approach 2 (Aurora_2: Retrieval + Groq LLM)

This version uses a Groq-hosted LLM and a stricter retrieval pipeline:

1. Fetches `/messages`.
2. Ranks messages by TF-IDF similarity to the question.
3. Builds a compact context with the top-k messages.
4. Calls the Groq LLM with a “you must only answer from these messages” system prompt.
5. Returns the model’s answer in:

```json
{ "answer": "..." }
```

### 1. Set up and install

From the repo root:

```bash
cd Aurora_2

python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Groq API key

The LLM client reads `GROQ_API_KEY` from the environment.

In the same terminal where you’ll run uvicorn:

```bash
# Windows PowerShell
$env:GROQ_API_KEY = "gsk_...your_full_secret_here..."

# macOS / Linux
# export GROQ_API_KEY="gsk_...your_full_secret_here..."
```

### 3. Run the API server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### 4. Example queries

```bash
curl "http://localhost:8080/ask?q=What%20did%20the%20member%20say%20about%20the%20car%20service%3F"

curl "http://localhost:8080/ask?q=Which%20restaurant%20did%20Fatima%20El-Tahir%20request%20a%20table%20at%3F"

curl "http://localhost:8080/ask?q=When%20is%20Layla%20planning%20her%20trip%20to%20London%3F"
```

Example answers produced by this LLM approach:

- **Car service**

  > Several members, including Vikram Desai and Fatima El-Tahir, complimented the car service as “impeccable”, while Thiago Monteiro expressed dissatisfaction with its timing and quality…

- **Fatima restaurant request**

  > Fatima El-Tahir requested tables at several restaurants, including Nobu, Le Bernardin, The French Laundry, Eleven Madison Park, a top-rated sushi place in Tokyo, and The Ivy in Chelsea.

- **Layla’s London trip**

  > Layla’s trip to London is planned for next month, and she needs a suite at Claridge’s for five nights starting Monday.

As with the TF-IDF approach, if the messages lack the necessary information, the system is instructed to say so explicitly.

---

## API Contract

Both implementations share the same API contract.

### `GET /ask`

- **Query parameter:**  
  `q` – natural-language question  

- **Response:**

```json
{
  "answer": "..."
}
```

Assignment-style usage:

```bash
curl "http://localhost:8080/ask?q=When%20is%20Layla%20planning%20her%20trip%20to%20London%3F"
```

---

## Design Notes (Bonus 1)

### Approach 1 – Rule-based + TF-IDF (`Aurora/`)

**Idea**

- Use TF-IDF to find the most relevant messages.
- For specific question templates, use custom extractors:
  - trips: detect person + city + dates (“next Friday”, “November 15”),
  - cars: extract cardinal numbers/words from sentences mentioning “car(s)”,
  - restaurants: extract proper-noun restaurant names after words like “favorite” / “table”.
- For everything else, fall back to an extractive answer: return the most relevant sentence.

**Pros**

- Cheap and fast, no LLM cost.
- Deterministic behaviour and easy to unit test.
- Strong for well-structured questions the extractors know about.

**Cons**

- Fragile to phrasing. If the question wording changes (“ride” vs “car service”), rules might miss it.
- Hard to extend to new question types without adding more code.
- More effort to keep date parsing, name recognition, and heuristics correct and non-brittle.

### Approach 2 – Retrieval + Groq LLM (`Aurora_2/`)

**Idea**

- Use the same `/messages` API and TF-IDF retrieval, but delegate answer generation and paraphrasing to an LLM, constrained by a strict system prompt (“only answer using these messages; if you’re not sure, say you don’t know.”).

**Pros**

- Handles a much wider variety of phrasings and follow-up questions.
- Produces natural, summarized answers that can combine information across multiple messages.
- Easier to support new question types without modifying code.

**Cons**

- Requires an external LLM provider (Groq) and an API key.
- More expensive per request than the pure TF-IDF approach.
- Needs careful prompting to avoid hallucinations; still possible when the data is thin.

### Why keep both?

The two sub-projects illustrate different trade-offs:

- `Aurora` is closer to an old-school IR system with handcrafted logic.  
- `Aurora_2` is a modern retrieval-augmented generation (RAG) service.

In a real product, a hybrid of both (rules for critical domains, LLM for everything else) is attractive.

---

## Data Insights & Anomalies (Bonus 2)

While exploring the `/messages` data, a few interesting patterns and issues show up:

### Encoding glitches

Some texts contain characters like `impeccableâthank` where a UTF-8 apostrophe appears corrupted. This suggests an encoding mismatch in the upstream system. The QA model still works but answers may look cleaner than the raw text.

### Multiple perspectives on car service

- Several members praise the car service as “impeccable” and efficient.
- At least one member (Thiago Monteiro) complains about timing and quality.

The model’s summarized answer reflects both positive and negative experiences.

### Rich restaurant preferences

Messages include high-end restaurants across multiple cities:

- New York: Eleven Madison Park, Le Bernardin.
- California: The French Laundry.
- London: The Ivy (Chelsea).
- Japan: top-rated sushi place in Tokyo.
- Global: Nobu.

Members often ask for “a table for four”, indicating small groups rather than very large events.

### Travel patterns and temporal expressions

- Requests like “Arrange a Gulfstream for a quick trip to Dubai next Friday” mix concrete destinations with relative dates.
- The code resolves expressions like “this Friday” / “next Friday” into actual dates (in `America/New_York` timezone) for the rule-based version.

### Names in the spec vs. names in data

The problem statement mentions members like Layla, Vikram Desai, and Amira.  
Some of these names appear prominently in the dataset; others appear little or not at all.  
The system is deliberately designed to say:

> “I don’t have that information in the messages”

when the underlying data doesn’t support an answer, rather than hallucinating details.

---

## Possible Improvements

If this were extended further, the next steps might include:

- **Better entity resolution** – resolve users by `user_id` as well as name strings; handle nicknames and spelling variations.
- **Richer retrieval** – mix TF-IDF with dense embeddings for better recall across paraphrases.
- **Per-member profiles** – cache a per-member summary (“profile card”) that the LLM can use as additional context for follow-up questions.
- **Evaluation harness** – a small QA test set with expected answers to track regressions across changes.

---

## Deployment

Locally, both apps run with:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

The LLM version is designed to be easily deployed on any service that can:

- install `requirements.txt`, and  
- run the command above,  
- with `GROQ_API_KEY` set as an environment variable.

Once deployed, the public URL looks like:

```text
GET https://<your-service>/ask?q=What%20did%20the%20member%20say%20about%20the%20car%20service%3F
```

returning JSON:

```json
{ "answer": "..." }
```

You can plug that into a UI, script, or other services as required.
