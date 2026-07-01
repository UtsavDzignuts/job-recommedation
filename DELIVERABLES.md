# KRA-KPA Job Board Platform — AI Intelligence Layer: Deliverables

---

## 1. Updated FastAPI Routes

All AI Intelligence Layer endpoints are registered via a consolidated `ai_router` in `app/ai/routes/__init__.py` and mounted at the application root (no prefix).

| Method | Endpoint | Module | Description |
|--------|----------|--------|-------------|
| `GET` | `/health` | `app/main.py` | System health check |
| `GET` | `/ask-ai?query=...` | `app/ai/routes/rag.py` | RAG-powered natural language Q&A |
| `POST` | `/recommend` | `app/ai/routes/recommend.py` | AI job recommendations from resume text |
| `POST` | `/improve-description` | `app/ai/routes/improve.py` | Rewrite job descriptions in 3 styles |
| `POST` | `/agent/task` | `app/ai/routes/agent.py` | Multi-step AI agent task execution |
| `POST` | `/sync/full` | `app/ai/routes/sync.py` | Full re-sync of embeddings to vector DB |

### Route Architecture

- **Dependency Injection**: Each route uses FastAPI `Depends()` to inject configured service instances (RAGEngine, RecommendationEngine, DescriptionImprover, AIAgentExecutor).
- **Validation**: Pydantic v2 models handle request/response validation with automatic 422 responses.
- **Error Handling**: Consistent error format `{error, message, details}` with status codes 400, 422, 503.
- **CORS**: Configured via middleware to allow all origins for development.

---

## 2. Vector Embeddings Storage Configuration

### Provider: ChromaDB

| Setting | Default | Description |
|---------|---------|-------------|
| `VECTOR_DB_PROVIDER` | `chromadb` | Vector database provider |
| `VECTOR_DB_URL` | `http://localhost:8000` | ChromaDB server connection URL |
| `VECTOR_DB_API_KEY` | *(optional)* | API key for Chroma Cloud authentication |
| `CHROMA_TENANT` | *(optional)* | Chroma Cloud tenant ID |
| `CHROMA_DATABASE` | *(optional)* | Chroma Cloud database name |

### Collections

| Collection | Source Entity | Embedding Content | Key Metadata |
|---|---|---|---|
| `job_posts` | Job Post | title + description + requirements + location + company | entity_id, title, location, company |
| `companies` | Company | name + description + industry | entity_id, name, industry |
| `candidates` | Candidate | name + skills + experience + bio + education | entity_id, name, skills |

### Storage Implementation

- **Module**: `app/ai/vectorstore/chromadb_store.py`
- **Interface**: Abstract `VectorStoreInterface` (`app/ai/vectorstore/__init__.py`) with operations: `upsert`, `search`, `delete`, `health_check`
- **Factory Pattern**: `app/ai/vectorstore/factory.py` creates the appropriate store based on `VECTOR_DB_PROVIDER`
- **Connection Modes**:
  - **Chroma Cloud**: Uses `chromadb.CloudClient` when api_key, tenant, and database are all provided
  - **Self-hosted HTTP**: Uses `chromadb.HttpClient` for remote ChromaDB servers
- **Similarity Metric**: Cosine distance (converted to similarity score via `1 - distance`)
- **Score Filtering**: Configurable minimum relevance threshold (default 0.5)

### Embedding Generation

- **Module**: `app/ai/embedding_service.py`
- **Providers**: OpenAI (`text-embedding-ada-002`) or Google Gemini (`models/gemini-embedding-001`)
- **Batch Support**: `generate_embedding()` for single texts, `generate_embeddings_batch()` for bulk operations
- **Retry Logic**: 3 retries with exponential backoff (1s → 2s → 4s) for transient errors (429, 500, 502, 503, 504)

---

## 3. LLM and LangChain Pipeline

### LLM Provider Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_PROVIDER` | `google` | Provider: `openai` or `google` |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-ada-002` | OpenAI embedding model |
| `GOOGLE_API_KEY` | — | Google Gemini API key |
| `GOOGLE_CHAT_MODEL` | `gemini-2.0-flash` | Gemini chat model |
| `GOOGLE_EMBEDDING_MODEL` | `models/gemini-embedding-001` | Gemini embedding model |

### LangChain Components Used

| Component | Library | Usage |
|-----------|---------|-------|
| `ChatOpenAI` | `langchain-openai` | Chat completions for RAG, recommendations, descriptions |
| `ChatGoogleGenerativeAI` | `langchain-google-genai` | Gemini chat completions (alternative provider) |
| `OpenAIEmbeddings` | `langchain-openai` | Text embedding generation |
| `GoogleGenerativeAIEmbeddings` | `langchain-google-genai` | Gemini embedding generation |
| `AgentExecutor` | `langchain-classic` | ReAct agent orchestration |
| `create_react_agent` | `langchain-classic` | Agent creation with ReAct prompt |
| `PromptTemplate` | `langchain-core` | Prompt template rendering |
| `@tool` decorator | `langchain-core` | Defining LangChain-compatible tools |

### Pipeline Architecture

```
User Input → EmbeddingService (LangChain Embeddings) → Vector Search (ChromaDB)
     ↓
Retrieved Context → PromptTemplateManager (YAML) → Rendered Prompt
     ↓
Circuit Breaker → LLM (LangChain Chat Model) → Structured Response
```

### Resilience Patterns

- **Circuit Breaker** (`app/ai/circuit_breaker.py`): Closed → Open (5 failures/60s) → Half-Open (30s cooldown)
- **Retry with Backoff** (`app/ai/retry.py`): Async decorator with configurable max retries, exponential backoff
- **LLM Factory** (`app/ai/llm_factory.py`): Abstraction layer that creates the correct LangChain chat model based on `LLM_PROVIDER`

---

## 4. Working RAG API

### Endpoint: `GET /ask-ai?query=...`

### RAG Pipeline (Module: `app/ai/rag_engine.py`)

| Step | Operation | Component |
|------|-----------|-----------|
| 1 | Embed user query | `EmbeddingService.generate_embedding()` |
| 2 | Similarity search across 3 collections | `VectorStoreInterface.search()` |
| 3 | Filter by min relevance score (0.5) | ChromaDB cosine distance conversion |
| 4 | Build context from top-K documents | `RAGEngine._build_context()` |
| 5 | Render prompt template with context | `PromptTemplateManager.render("rag_answer")` |
| 6 | LLM completion via circuit breaker | `CircuitBreaker.call()` → ChatOpenAI/Gemini |
| 7 | Extract sources and format response | `AskAIResponse` with answer + sources |

### Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `RAG_TOP_K` | `10` | Max documents to retrieve |
| `RAG_MIN_RELEVANCE_THRESHOLD` | `0.5` | Minimum similarity score |
| `RAG_MAX_QUERY_LENGTH` | `1000` | Max query length (characters) |

### Response Format

```json
{
  "answer": "Based on available listings...",
  "sources": [
    {
      "entity_type": "job_post",
      "entity_id": "job_1",
      "text_snippet": "Senior Python Developer...",
      "relevance_score": 0.92
    }
  ],
  "query": "What Python jobs are available?"
}
```

### Fallback Behavior

When the LLM is unavailable (circuit breaker open or timeout), the RAG engine returns a structured fallback answer built directly from the vector search results, categorized by entity type (jobs, companies, candidates) with match percentages.

---

## 5. AI Recommendation API

### Endpoint: `POST /recommend`

### Pipeline (Module: `app/ai/recommendation_engine.py`)

| Step | Operation |
|------|-----------|
| 1 | Embed resume text using `EmbeddingService` |
| 2 | Search `job_posts` collection for top matches |
| 3 | Format matched jobs for LLM evaluation |
| 4 | Render `job_recommendation` prompt template |
| 5 | LLM ranks and scores matches (up to 5 results) |
| 6 | Return structured recommendations with confidence scores |

### Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `RECOMMEND_TOP_K` | `10` | Vector search candidates to retrieve |
| `RECOMMEND_MAX_RESULTS` | `5` | Max recommendations returned |

### Request/Response

**Request:**
```json
{
  "resume_text": "5 years Python, FastAPI, machine learning, AWS"
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "job_title": "Senior ML Engineer",
      "job_id": "456",
      "match_reason": "Strong Python and ML background matches requirements",
      "confidence_score": 0.93
    }
  ],
  "message": null
}
```

### Validation

- `resume_text`: Required, 1–10,000 characters
- Returns HTTP 422 for invalid input, HTTP 503 if LLM unavailable

---

## 6. AI Agent Implementation and Example Prompts

### Architecture

- **Module**: `app/ai/agent/executor.py`
- **Pattern**: ReAct (Reason + Act) loop via LangChain's `AgentExecutor`
- **Max Steps**: 10 (configurable via `AGENT_MAX_STEPS`)
- **LLM**: Same provider as other features (OpenAI or Gemini)

### Available Tools (Module: `app/ai/agent/tools.py`)

| Tool | Function | Description |
|------|----------|-------------|
| `api_query_tool` | Queries FastAPI endpoints | Fetches jobs, companies, or candidates from the platform API |
| `vector_search_tool` | Semantic similarity search | Searches all vector DB collections using embeddings |
| `llm_reasoning_tool` | LLM analysis/summarization | Calls the LLM for complex reasoning, comparison, ranking |

### ReAct Reasoning Loop

```
Question → Think → Act (choose tool) → Observe (tool output) → Think → ... → Final Answer
```

The agent:
1. Analyzes the task and decides which tool to use first
2. Invokes the tool with appropriate input
3. Evaluates the result
4. Repeats (up to 10 steps) until task is complete
5. If a tool fails, tries an alternative approach

### Example Prompts

| Prompt | Expected Agent Behavior |
|--------|------------------------|
| `"Find all remote Python jobs and summarize the requirements"` | vector_search → llm_reasoning |
| `"Find the top 3 jobs related to cloud computing and summarize them"` | vector_search → llm_reasoning |
| `"Compare candidates for a senior backend role"` | vector_search (candidates) → vector_search (jobs) → llm_reasoning |
| `"Which companies are hiring for data science?"` | vector_search → api_query → llm_reasoning |
| `"List all jobs and tell me which ones need Python"` | api_query → llm_reasoning |

### Endpoint: `POST /agent/task`

**Request:**
```json
{
  "task": "Find all remote Python jobs and summarize the requirements"
}
```

**Response:**
```json
{
  "answer": "I found 3 Python developer positions...",
  "steps": [
    {
      "tool_name": "vector_search_tool",
      "input": {"query": "Python developer"},
      "output": "Found 3 matching job posts...",
      "reasoning": "Searching for Python developer positions"
    }
  ],
  "completed": true,
  "message": null
}
```

---

## 7. Postman Collection

**File**: `backend/docs/ai-intelligence-layer.postman_collection.json`

### Collection Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `base_url` | `http://localhost:8000` | Base URL of the FastAPI server |
| `api_key` | *(your key)* | Bearer token for authentication |

### Included Requests

| # | Request Name | Method | Endpoint |
|---|---|---|---|
| 1 | Ask AI (RAG Question Answering) | GET | `/ask-ai?query=List top remote Python jobs for a fresher` |
| 2 | Recommend Jobs | POST | `/recommend` |
| 3 | Improve Job Description | POST | `/improve-description` |
| 4 | AI Agent Task | POST | `/agent/task` |
| 5 | Full Sync | POST | `/sync/full` |

Each request includes:
- Pre-configured headers (Content-Type, Authorization)
- Sample request bodies with realistic data
- Expected success and error response examples
- Detailed endpoint documentation

### Import Instructions

1. Open Postman
2. Click **Import** → **File**
3. Select `backend/docs/ai-intelligence-layer.postman_collection.json`
4. Set the `base_url` variable to your running server address
5. Set the `api_key` if authentication is enabled

---

## 8. Documentation

### Available Documentation Files

| File | Location | Content |
|------|----------|---------|
| `README.md` | Project root | Full project overview, setup guide, architecture, API examples |
| `AI_INTELLIGENCE_LAYER.md` | `backend/docs/` | Detailed system architecture, Mermaid diagrams, tool versions, all prompt templates |
| `DELIVERABLES.md` | Project root | This deliverables summary document |

### Documentation Covers

- System architecture (Mermaid diagrams)
- Component responsibilities and module mapping
- Data flow diagrams (Query → Response)
- Vector database collections and metadata schema
- All 6 prompt template specifications (variables, behavior, usage scenarios)
- Error handling strategy and retry logic
- Circuit breaker states and configuration
- Environment variable reference

---

## 9. System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Next.js Frontend                         │
│  (Ask AI │ Recommend │ Improve │ Agent │ Sync)              │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP
┌─────────────────────────▼───────────────────────────────────┐
│                   FastAPI Backend                            │
│  ┌──────────┐ ┌────────────────┐ ┌───────────────────────┐ │
│  │RAGEngine │ │RecommendEngine │ │DescriptionImprover    │ │
│  └────┬─────┘ └───────┬────────┘ └──────────┬────────────┘ │
│       │                │                     │              │
│  ┌────▼────────────────▼─────────────────────▼────────────┐ │
│  │   EmbeddingService + PromptManager + CircuitBreaker     │ │
│  └────┬──────────────────────────────┬────────────────────┘ │
│       │                              │                      │
│  ┌────▼────────┐           ┌────────▼─────────┐            │
│  │ ChromaDB    │           │ OpenAI / Gemini   │            │
│  │ (vectors)   │           │ (LLM + embeddings)│            │
│  └─────────────┘           └──────────────────┘            │
│                                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │ AIAgentExecutor (ReAct loop via LangChain)        │      │
│  │  Tools: api_query │ vector_search │ llm_reasoning │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Module | Responsibility |
|---|---|---|
| RAGEngine | `app/ai/rag_engine.py` | Retrieval-augmented generation Q&A |
| RecommendationEngine | `app/ai/recommendation_engine.py` | Resume-to-job matching with LLM ranking |
| DescriptionImprover | `app/ai/description_improver.py` | Job description rewriting (3 modes) |
| AIAgentExecutor | `app/ai/agent/executor.py` | Autonomous multi-step reasoning agent |
| EmbeddingService | `app/ai/embedding_service.py` | Text-to-vector conversion with retry |
| SyncService | `app/ai/sync_service.py` | PostgreSQL → ChromaDB data sync |
| PromptTemplateManager | `app/ai/prompt_manager.py` | YAML-based prompt loading and rendering |
| CircuitBreaker | `app/ai/circuit_breaker.py` | LLM call protection (closed/open/half-open) |
| VectorStoreInterface | `app/ai/vectorstore/__init__.py` | Abstract vector DB operations |
| ChromaDBStore | `app/ai/vectorstore/chromadb_store.py` | ChromaDB-specific implementation |
| LLMFactory | `app/ai/llm_factory.py` | Provider-agnostic LLM instantiation |

### Design Patterns

- **Factory Pattern**: Vector store and LLM creation abstracted behind factories
- **Circuit Breaker**: Protects against cascading LLM failures
- **Retry with Exponential Backoff**: Handles transient API errors
- **Dependency Injection**: FastAPI `Depends()` for service composition
- **Strategy Pattern**: Prompt templates selected by mode/task type
- **ReAct Pattern**: Agent reasoning loop (Think → Act → Observe)

---

## 10. Tools Used

### Backend Stack

| Tool / Library | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| FastAPI | >= 0.104.0 | Async REST API framework |
| Uvicorn | >= 0.24.0 | ASGI server |
| Pydantic | >= 2.5.0 | Request/response validation |
| pydantic-settings | >= 2.1.0 | Environment configuration |
| SQLAlchemy | >= 2.0.0 | ORM and database toolkit |
| asyncpg | >= 0.29.0 | Async PostgreSQL driver |
| Alembic | >= 1.13.0 | Database migrations |
| ChromaDB | >= 0.4.22 | Vector database |
| LangChain | >= 0.1.0 | LLM orchestration framework |
| langchain-openai | >= 0.0.5 | OpenAI integration |
| langchain-community | >= 0.0.10 | Community integrations |
| langchain-google-genai | >= 1.0.0 | Google Gemini integration |
| langchain-classic | >= 0.0.1 | Agent framework (ReAct) |
| OpenAI SDK | >= 1.6.0 | LLM and embeddings client |
| google-generativeai | >= 0.3.0 | Gemini SDK |
| PyYAML | >= 6.0.1 | Prompt template parsing |
| httpx | >= 0.25.0 | Async HTTP client |
| python-dotenv | >= 1.0.0 | Environment file loading |

### Frontend Stack

| Tool / Library | Purpose |
|---|---|
| Next.js 14 | React framework with App Router |
| React 18 | UI library |
| TypeScript | Type safety |
| Tailwind CSS | Utility-first styling |
| react-markdown | Render AI responses |

### Testing

| Tool | Purpose |
|---|---|
| pytest >= 7.4.0 | Test framework |
| pytest-asyncio >= 0.23.0 | Async test support |
| Hypothesis >= 6.92.0 | Property-based testing |

### Infrastructure

| Tool | Purpose |
|---|---|
| Docker | Containerization (backend + ChromaDB) |
| Render.com | Cloud deployment (Blueprint in `render.yaml`) |
| Alembic | Database schema migrations |

---

## 11. How to Run Locally

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- PostgreSQL (with job board data)
- OpenAI API key or Google Gemini API key

### Step 1: Clone and Enter Project

```bash
git clone <repository-url>
cd kra-kpa
```

### Step 2: Start ChromaDB (Vector Database)

```bash
# Option A: Run directly
chroma run --path ./chroma_data --port 8001

# Option B: Docker
docker run -p 8001:8000 chromadb/chroma:latest

# Option C: Using the included Dockerfile
docker build -f backend/Dockerfile.chroma -t kra-kpa-chroma .
docker run -p 8001:8000 kra-kpa-chroma
```

### Step 3: Set Up Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — set OPENAI_API_KEY (or GOOGLE_API_KEY) and VECTOR_DB_URL

# Run database migrations
alembic upgrade head

# Start the server
uvicorn app.main:app --reload --port 8000
```

### Step 4: Sync Embeddings

```bash
# Populate vector database with sample data
curl -X POST http://localhost:8000/sync/full
```

### Step 5: Set Up Frontend

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:8000" > .env.local

# Start development server
npm run dev
```

### Step 6: Verify

- Backend API: http://localhost:8000/health
- Frontend: http://localhost:3000
- API docs (Swagger): http://localhost:8000/docs
- Test a query: `curl "http://localhost:8000/ask-ai?query=What+Python+jobs+are+available?"`

### Running Tests

```bash
cd backend

# All tests
pytest

# Verbose
pytest -v

# Specific modules
pytest tests/test_rag_engine.py
pytest tests/test_agent_executor.py
pytest tests/test_recommendation_engine.py
```

### Docker Deployment (Render.com)

The project includes `render.yaml` Blueprint for one-click deployment:

```bash
# Services deployed:
# 1. kra-kpa-backend (FastAPI, Dockerfile)
# 2. ChromaDB (Dockerfile.chroma) — or use Chroma Cloud

# Set these secrets in Render dashboard:
# - OPENAI_API_KEY
# - VECTOR_DB_URL
# - VECTOR_DB_API_KEY (if using Chroma Cloud)
# - CHROMA_TENANT (if using Chroma Cloud)
# - CHROMA_DATABASE (if using Chroma Cloud)
```

---

## Summary

| # | Deliverable | Status | Location |
|---|---|---|---|
| 1 | Updated FastAPI Routes | ✅ Complete | `app/ai/routes/` (5 route modules) |
| 2 | Vector Embeddings Storage Configuration | ✅ Complete | `app/ai/vectorstore/`, `app/ai/embedding_service.py` |
| 3 | LLM and LangChain Pipeline | ✅ Complete | `app/ai/llm_factory.py`, `app/ai/config.py` |
| 4 | Working RAG API | ✅ Complete | `app/ai/rag_engine.py`, `GET /ask-ai` |
| 5 | AI Recommendation API | ✅ Complete | `app/ai/recommendation_engine.py`, `POST /recommend` |
| 6 | AI Agent Implementation | ✅ Complete | `app/ai/agent/executor.py`, `app/ai/agent/tools.py` |
| 7 | Postman Collection | ✅ Complete | `backend/docs/ai-intelligence-layer.postman_collection.json` |
| 8 | Documentation | ✅ Complete | `README.md`, `backend/docs/AI_INTELLIGENCE_LAYER.md` |
| 9 | System Architecture | ✅ Complete | Documented with diagrams in README and docs |
| 10 | Tools Used | ✅ Complete | Listed in `requirements.txt` and documented above |
| 11 | How to Run Locally | ✅ Complete | Step-by-step instructions above |
