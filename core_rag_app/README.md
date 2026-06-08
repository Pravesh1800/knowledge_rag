# Evidence Mesh

This is the generic core retrieval copy of the original application.

It keeps:

- project creation
- document upload and manifest management
- PDF/page rendering and spreadsheet ingestion
- page/card indexing
- knowledge-graph generation
- clusters, domains, and domain relationships
- graph-aware search
- grounded chat with citations

It removes:

- tender-specific report generation
- legal assessment agent
- commercial strategy agent
- financial bonds agent
- pre-bid query agent
- pre-qualification requirements agent
- playbooks and domain-specific report UI

## Setup

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create `.env` from `.env.example` or add your own:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=google/gemini-3.1-flash-lite
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_TIMEOUT_SECONDS=120
OPENROUTER_MAX_RETRIES=1
```

## Run

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8021
```

Open:

```text
http://127.0.0.1:8021
```

## CLI Usage

Ingest documents:

```powershell
python ingest.py add C:\path\to\documents
```

Build the page/card index:

```powershell
python indexer.py
```

Build clusters, domains, and relationships:

```powershell
python knowledge_graph.py
```

Search:

```powershell
python searcher.py "your question or card"
```

## Data Layout

Runtime data is stored under:

```text
projects/<project_id>/
  documents/
  indexes/
  logs/
  uploads/
```

Core index files:

```text
indexes/card_index.json
indexes/knowledge_graph.json
indexes/search_results/
```

## Verify

```powershell
python -m py_compile app.py ingest.py indexer.py knowledge_graph.py searcher.py
```



