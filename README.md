# KRA-KPA Job Board Platform — AI Intelligence Layer

An AI-powered Job Board Platform that combines semantic search, natural language Q&A, intelligent job recommendations, description improvement, and an autonomous reasoning agent — all built on top of a FastAPI backend with a Next.js frontend.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Frontend Pages](#frontend-pages)
- [How It Works](#how-it-works)
- [Testing](#testing)
- [Deployment](#deployment)
- [Project Structure](#project-structure)

---

## Features

| Feature | Description |
|---------|-------------|
| 🤖 **Ask AI (RAG Q&A)** | Ask natural language questions about jobs, companies, and candidates. Answers are grounded in real data via retrieval-augmented generation. |
| 💼 **Job Recommendations** | Paste a resume and get AI-ranked job recommendations with confidence scores and match explanations. |
| ✍️ **Improve Description** | Rewrite job descriptions in three styles: short & crisp, detailed & formal, or marketing-oriented. |
| 🧠 **AI Agent** | Give complex tasks to an autonomous agent that reasons step-by-step, invoking tools (API queries, vector search, LLM reasoning) to produce results. |
| 🔄 **Embedding Sync** | Synchronize job board data (jobs, companies, candidates) from PostgreSQL into the vector database for up-to-date AI features. |

---

## Tech Stack

### Backend

| Technology | Purpose |
|------------|---------|
| Python 3.10+ | Runtime |
| FastAPI | Async REST API framework |
| SQLAlchemy + asyncpg | ORM and async PostgreSQL driver |
| Alembic | Database migrations |
| LangChain | LLM orchestration and agent framework |
| OpenAI / Google Gemini | LLM and embedding providers |
| ChromaDB | Vector database for semantic search |
| Pydantic v2 | Request/response validation |

### Frontend

| Technology | Purpose |
|------------|---------|
| Next.js 14 | React framework with App Router |
| React 18 | UI library |
| TypeScript | Type safety |
| Tailwind CSS | Utility-first styling |
| react-markdown | Render AI responses with markdown |

### Infrastructure

| Technology | Purpose |
|------------|---------|
| Docker | Containerization |
| Render.com | Cloud deployment (Blueprint) |
| ChromaDB (standalone) | Vector database server |

---

## Architecture

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
│  │         EmbeddingService + PromptManager               │ │
│  │              + CircuitBreaker                           │ │
│  └────┬──────────────────────────────┬────────────────────┘ │
│       │                              │                      │
│  ┌────▼────────┐           ┌────────▼─────────┐            │
│  │ ChromaDB    │           │ OpenAI / Gemini   │            │
│  │ (vectors)   │           │ (LLM + embeddings)│            │
│  └─────────────┘           └──────────────────┘            │
│                                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │ AIAgentExecutor (ReAct loop)                      │      │
│  │  Tools: api_query │ vector_search │ llm_reasoning │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Key Architectural Patterns

- **Circuit Breaker** — Protects LLM calls from cascading failures (closed → open → half-open states)
- **Retry with Backoff** — Transient API errors are retried up to 3 times with exponential backoff
- **YAML Prompt Templates** — All prompts are externalized as YAML files for easy editing
- **Vector Store Abstraction** — Factory pattern allows swapping vector DB implementations
- **ReAct Agent** — Multi-step reasoning agent that autonomously uses tools to solve tasks

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- PostgreSQL (running with job board data)
- An OpenAI API key or Google Gemini API key

### Clone the Repository

```bash
git clone <repository-url>
cd kra-kpa
```

---

## Environment Variables

### Backend (`backend/.env`)

```env
# LLM Provider: 'google' or 'openai'
LLM_PROVIDER=openai

# Google Gemini (if using Google)
GOOGLE_API_KEY=your-google-api-key
GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001
GOOGLE_CHAT_MODEL=gemini-2.0-flash

# OpenAI (if using OpenAI)
OPENAI_API_KEY=your-openai-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_CHAT_MODEL=gpt-4o-mini

# Vector Database
VECTOR_DB_PROVIDER=chromadb
VECTOR_DB_URL=http://localhost:8001
```

### Frontend (`frontend/.env.local`)

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

---

## Running the Application

### 1. Start ChromaDB (Vector Database)

```bash
# Option A: Run directly
chroma run --path ./chroma_data --port 8001

# Option B: Run via Docker
docker run -p 8001:8000 chromadb/chroma:latest
```

### 2. Set Up the Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Seed the vector database with sample data (optional)
python seed_data.py

# Start the backend server
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

### 3. Set Up the Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:3000`.

### 4. Sync Embeddings

After the backend is running, populate the vector database:

```bash
curl -X POST http://localhost:8000/sync/full
```

Or use the **Sync Embeddings** page in the frontend.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/ask-ai?query=...` | RAG-powered natural language Q&A |
| `POST` | `/recommend` | Job recommendations from resume text |
| `POST` | `/improve-description` | Rewrite a job description in a chosen style |
| `POST` | `/agent/task` | Execute a complex task via the AI agent |
| `POST` | `/sync/full` | Full re-sync of embeddings from PostgreSQL |

### Example: Ask AI

```bash
curl "http://localhost:8000/ask-ai?query=What%20Python%20jobs%20are%20available?"
```

**Response:**
```json
{
  "answer": "Based on available listings, there are several Python positions...",
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

### Example: Job Recommendations

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "5 years Python, FastAPI, machine learning, AWS"}'
```

### Example: Improve Description

```bash
curl -X POST http://localhost:8000/improve-description \
  -H "Content-Type: application/json" \
  -d '{"description": "We need a Python dev...", "mode": "short_and_crisp"}'
```

Modes: `short_and_crisp`, `detailed_and_formal`, `marketing_oriented`

### Example: AI Agent

```bash
curl -X POST http://localhost:8000/agent/task \
  -H "Content-Type: application/json" \
  -d '{"task": "Find all remote Python jobs and summarize the requirements"}'
```

---

## Frontend Pages

| Page | Route | Description |
|------|-------|-------------|
| Dashboard | `/` | Overview of all AI features with quick stats |
| Ask AI | `/ask-ai` | Chat interface for natural language questions |
| Recommendations | `/recommend` | Paste resume, get matched jobs |
| Improve Description | `/improve-description` | Rewrite job descriptions with style options |
| AI Agent | `/agent` | Submit complex tasks for autonomous execution |
| Sync | `/sync` | Trigger embedding synchronization |

---

## How It Works

### Ask AI (RAG Pipeline)

1. User submits a natural language question
2. The query is embedded using OpenAI/Gemini embeddings
3. ChromaDB performs similarity search across job posts, companies, and candidates
4. Top relevant documents are assembled as context
5. A prompt template is rendered with the query + context
6. The LLM generates an answer grounded in the retrieved documents
7. The response includes the answer and source references with relevance scores

### Job Recommendations

1. User submits their resume text
2. Resume is embedded and compared against all job post embeddings
3. Top matching jobs are retrieved from ChromaDB
4. The LLM ranks matches and explains why each job is a good fit
5. Returns up to 5 recommendations with confidence scores (0–1)

### Description Improvement

1. User provides a raw job description and selects a style
2. The appropriate YAML prompt template is loaded for the chosen mode
3. The LLM rewrites the description according to the style instructions
4. Returns the polished description

### AI Agent (ReAct Reasoning)

1. User describes a complex task
2. The agent enters a think → act → observe loop:
   - **Think**: Decides what to do next
   - **Act**: Calls one of three tools (API query, vector search, LLM reasoning)
   - **Observe**: Evaluates the tool's output
3. Repeats up to 10 steps until the task is complete
4. Returns the final answer along with all intermediate reasoning steps

### Embedding Sync

1. Reads all entities (jobs, companies, candidates) from PostgreSQL
2. Generates embeddings for each entity's text content
3. Upserts embeddings into ChromaDB collections
4. Tracks sync status in the `embedding_sync_status` table for retry handling

---

## Testing

```bash
cd backend

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test files
pytest tests/test_rag_engine.py
pytest tests/test_recommendation_engine.py
pytest tests/test_agent_executor.py
```

The test suite includes unit tests for all AI components and property-based tests using Hypothesis.

---

## Deployment

The project includes a Render Blueprint (`backend/render.yaml`) for one-click deployment:

```bash
# Deploys:
# 1. FastAPI backend (Docker)
# 2. ChromaDB vector database (Docker)
```

Set `OPENAI_API_KEY` and `VECTOR_DB_URL` manually in the Render dashboard.

---

## Project Structure

```
kra-kpa/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI entry point
│   │   ├── ai/
│   │   │   ├── rag_engine.py          # RAG Q&A engine
│   │   │   ├── recommendation_engine.py # Job matching engine
│   │   │   ├── description_improver.py  # Description rewriting
│   │   │   ├── embedding_service.py     # Embedding generation
│   │   │   ├── sync_service.py          # PostgreSQL → ChromaDB sync
│   │   │   ├── circuit_breaker.py       # LLM fault tolerance
│   │   │   ├── prompt_manager.py        # YAML prompt template loading
│   │   │   ├── llm_factory.py           # LLM provider abstraction
│   │   │   ├── agent/
│   │   │   │   ├── executor.py          # ReAct agent executor
│   │   │   │   └── tools.py            # Agent tools (api, vector, llm)
│   │   │   ├── routes/                  # API route handlers
│   │   │   └── vectorstore/             # Vector DB abstraction layer
│   │   └── prompts/                     # YAML prompt templates
│   ├── alembic/                         # Database migrations
│   ├── tests/                           # Test suite
│   ├── docs/                            # API documentation
│   ├── seed_data.py                     # Sample data seeder
│   ├── requirements.txt                 # Python dependencies
│   ├── Dockerfile                       # Backend container
│   └── Dockerfile.chroma               # ChromaDB container
├── frontend/
│   ├── app/                             # Next.js pages (App Router)
│   │   ├── ask-ai/                      # RAG Q&A page
│   │   ├── recommend/                   # Recommendations page
│   │   ├── improve-description/         # Description improvement page
│   │   ├── agent/                       # AI agent page
│   │   └── sync/                        # Sync management page
│   ├── components/                      # Shared React components
│   └── package.json                     # Node.js dependencies
└── README.md
```

---

## License

This project is for internal/educational use.
