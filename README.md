# Medium Article RAG Assistant

Individual RAG assignment: a FastAPI Retrieval-Augmented Generation assistant for the provided Medium articles dataset.

The system answers questions only from retrieved Medium article context stored in Pinecone. If the retrieved dataset context does not support the answer, the assistant must respond that it does not know based on the provided Medium articles data.

## Live Submission

- Live URL: add Vercel URL here after deployment
- GitHub URL: add public repository URL here after pushing

## RAG Configuration

These are the current hyperparameters returned by `GET /api/stats`:

```json
{
  "chunk_size": 700,
  "overlap_ratio": 0.15,
  "top_k": 6
}
```

Vector database:

- Pinecone index: `medium-rag`
- Pinecone namespace: `medium-full-700-015`
- Embedding model: `4UHRUIN-text-embedding-3-small`
- Embedding dimension: `1536`
- Chat model: `4UHRUIN-gpt-5-mini`

The full dataset was ingested into Pinecone:

- Articles: `7,682`
- Chunks/vectors: `16,427`

## API Endpoints

### `GET /api/stats`

Returns the RAG hyperparameters:

```json
{
  "chunk_size": 700,
  "overlap_ratio": 0.15,
  "top_k": 6
}
```

### `POST /api/prompt`

Request:

```json
{
  "question": "Your natural language question here"
}
```

Response:

```json
{
  "response": "Final natural language answer from the model.",
  "context": [
    {
      "article_id": "1234",
      "title": "Sample article title",
      "chunk": "article chunk retrieved",
      "score": 0.1234
    }
  ],
  "Augmented_prompt": {
    "System": "the system prompt used to query the chat model",
    "User": "the user prompt used to query the chat model"
  }
}
```

## Retrieval Workflow

1. The user question is converted into retrieval queries:
   - the original question,
   - a rule-based keyword query,
   - one GPT-generated search query used only for retrieval.
2. The retrieval queries are embedded with `4UHRUIN-text-embedding-3-small`.
3. Pinecone is queried in namespace `medium-full-700-015`.
4. Retrieved chunks are deduplicated by `article_id`, so the final context contains distinct articles.
5. The chat model receives the retrieved context and must answer only from that context.

The GPT-generated retrieval query is not treated as evidence. It is only a search string. The final answer must still be grounded in the retrieved Medium article chunks.

## Local Setup

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create `.env` from `.env.example`, then fill in API keys:

```powershell
Copy-Item .env.example .env
```

Run locally:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.index:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Ingestion

The full dataset is already uploaded to Pinecone namespace `medium-full-700-015`.

To preview chunking without API cost:

```powershell
.\.venv\Scripts\python.exe -m scripts.preview_dataset_chunks
```

To ingest a small test subset:

```powershell
.\.venv\Scripts\python.exe -m scripts.ingest --limit 100 --namespace medium-test-100
```

To ingest the full dataset:

```powershell
.\.venv\Scripts\python.exe -m scripts.ingest --limit 0 --batch-size 64 --namespace medium-full-700-015
```

Do not re-ingest the full dataset unless needed, because it spends embedding budget.

## Deployment

Deploy to Vercel and set these environment variables in the Vercel project:

```text
OPENAI_API_KEY
OPENAI_BASE_URL
CHAT_MODEL
EMBEDDING_MODEL
PINECONE_API_KEY
PINECONE_INDEX
PINECONE_NAMESPACE
CHUNK_SIZE
OVERLAP_RATIO
TOP_K
DRY_RUN
```

Use these values for the non-secret variables:

```text
OPENAI_BASE_URL=https://api.llmod.ai
CHAT_MODEL=4UHRUIN-gpt-5-mini
EMBEDDING_MODEL=4UHRUIN-text-embedding-3-small
PINECONE_INDEX=medium-rag
PINECONE_NAMESPACE=medium-full-700-015
CHUNK_SIZE=700
OVERLAP_RATIO=0.15
TOP_K=6
DRY_RUN=false
```

The CSV dataset is not required at runtime because the deployed app retrieves from Pinecone.

## Evaluation Summary

The final system was tested on a 12-question evaluation set covering:

- precise fact retrieval,
- multi-result topic listing,
- key idea summary extraction,
- recommendation with evidence-based justification,
- impossible questions that require refusal.

The fresh 12-question evaluation passed all 12 tests. One earlier ultra-hard query about the bubonic plague and AI remained a safe retrieval miss: the system refused instead of inventing an unsupported answer.
