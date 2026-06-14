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
LLM_PROVIDER=openrouter
LLM_API_KEY=
LLM_MODEL=google/gemini-3.1-flash-lite
LLM_MAP_MODEL=deepseek/deepseek-v4-flash
LLM_SEARCH_MODEL=deepseek/deepseek-v4-flash

OPENROUTER_API_KEY=your_openrouter_key_here
OPENAI_API_KEY=your_openai_key_here
```

You can also change the active provider, API key, and model names from the
Settings page in the UI.

## Run

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8021
```

Open:

```text
http://127.0.0.1:8021
```

## PostgreSQL Storage

PostgreSQL is the primary storage for documents, pages, cards, graph objects, search
runs, and build runs. JSON files under `projects/<project>/indexes` are compatibility
exports, not the source of truth.

```env
EVIDENCE_MESH_STORAGE=postgres
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/evidence_mesh
POSTGRES_CONNECT_TIMEOUT=5
EVIDENCE_MESH_PROJECT_ID=local
```

```powershell
python storage.py init-postgres
```

The app writes runtime data into:

- `documents`
- `pages`
- `cards`
- `clusters`
- `domains`
- `relationships`
- `search_runs`
- `graph_build_runs`

The typed columns make recovery and querying safer, while each row also keeps the full
source JSON in `payload` so schema changes do not drop data. For throwaway local
debugging only, `EVIDENCE_MESH_STORAGE=json` disables the required PostgreSQL checks.

## CLI Usage

Ingest documents:

```powershell
python ingest.py add C:\path\to\documents
```

Build the page/card index:

```powershell
python indexer.py
```

Build or refresh the semantic embedding sidecar used by hybrid retrieval:

```powershell
python embeddings.py build
```

Build or refresh typed entity and claim anchors:

```powershell
python entity_claims.py build
```

Build canonical entity aliases for deduplication and retrieval:

```powershell
python entity_canonicalizer.py build
```

For a cheap heuristic-only first pass:

```powershell
python entity_claims.py build --dry-run
```

You can also build embeddings after indexing in one pass:

```powershell
python indexer.py --build-embeddings --build-entity-claims
```

Build clusters, domains, and relationships:

```powershell
python knowledge_graph.py
```

Build or refresh precomputed domain/community summaries:

```powershell
python community_summaries.py build
```

For a heuristic-only summary pass:

```powershell
python community_summaries.py build --dry-run
```

You can also build summaries after the graph in one pass:

```powershell
python knowledge_graph.py --build-summaries
```

Search:

```powershell
python searcher.py "your question or card"
```

Search uses hybrid retrieval plus a second-stage reranker by default. Tune with:

```env
EVIDENCE_MESH_RERANK=1
EVIDENCE_MESH_RERANK_CANDIDATES=40
EVIDENCE_MESH_RERANK_MAX_TOKENS=1800
```

Chat automatically classifies each question into an adaptive retrieval mode:
`exact_lookup`, `multi_hop`, `comparison`, `global_summary`,
`contradiction_check`, `gap_analysis`, `risk_analysis`, `follow_up`, or
`general`. The API accepts an optional `query_mode` override; omit it or use
`auto` for automatic mode selection.

Relationship links are retrieval-active in graph-aware modes: strong domain and
card relationships can boost candidate scores, expand connected cards, and pull
related domains into traversal before final reranking.

Run an evaluation set:

```powershell
python evaluator.py eval_sets/example_retrieval_eval.json --max-hits 12
```

Evaluation sets are versioned JSON files with real questions, expected answer
points, expected evidence cards/pages, and cross-document expectations. Reports
are written under:

```text
indexes/eval_reports/
```

Each missed evidence item includes failure diagnosis, such as:

- no matching domain visited
- domain visited but cluster missed
- cluster visited but card missed
- card retrieved but page mismatch
- expected document never entered the search path

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
indexes/card_embeddings.json
indexes/entity_claim_index.json
indexes/canonical_entities.json
indexes/community_summaries.json
indexes/knowledge_graph.json
indexes/search_results/
indexes/eval_reports/
```

## Verify

```powershell
python -m py_compile app.py ingest.py indexer.py embeddings.py entity_claims.py entity_canonicalizer.py community_summaries.py query_modes.py reranker.py knowledge_graph.py searcher.py evaluator.py
```



