# Evidence Mesh Core RAG App

Evidence Mesh is a graph-aware RAG application for project-document question
answering. It turns PDFs, spreadsheets, text files, and image-based documents
into page-linked evidence cards, builds a domain and relationship graph, and
answers questions with cited source evidence.

## What It Does

- Creates isolated projects for document sets
- Uploads and manages source files
- Parses PDFs, spreadsheets, text files, and images
- Builds evidence cards with document and page metadata
- Extracts typed entities and atomic claims
- Builds canonical entity aliases for deduplication
- Builds clusters, domains, community summaries, and domain relationships
- Performs graph-aware hybrid retrieval
- Uses adaptive query modes for exact lookup, multi-hop, comparison, summary,
  contradiction, gap, risk, follow-up, and general questions
- Reranks retrieved evidence
- Generates grounded answers with document/page citations
- Runs retrieval evaluations with case-level failure diagnosis

## Current Benchmark Summary

Reported final test-folder evaluation:

| Metric | Score |
|---|---:|
| Evidence recall | 98.0% |

The 98.0% evidence-recall metric is benchmark-specific. It should be cited
with the benchmark context, not as a universal state-of-the-art claim.

## Research Paper

The full research write-up is in:

[docs/research_paper.md](docs/research_paper.md)

Architecture notes are in:

[docs/architecture.md](docs/architecture.md)

## Requirements

- Python 3.12 or newer
- An OpenRouter or OpenAI-compatible API key
- Optional: PostgreSQL for typed persistent storage

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Environment Setup

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Set your provider and model configuration:

```env
LLM_PROVIDER=openrouter
LLM_API_KEY=
LLM_MODEL=google/gemini-3.1-flash-lite
LLM_MAP_MODEL=deepseek/deepseek-v4-flash
LLM_SEARCH_MODEL=deepseek/deepseek-v4-flash

OPENROUTER_API_KEY=your_openrouter_key_here
OPENAI_API_KEY=your_openai_key_here
```

You can also edit provider and model settings from the UI Settings page.

## Run the Web App

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8021
```

Open the v2 UI:

```text
http://127.0.0.1:8021/v2
```

## Web Workflow

1. Create a project.
2. Upload source documents.
3. Build the evidence mesh.
4. Ask questions in Retrieval chat.
5. Inspect top evidence and search trace.
6. Use the Graph page to inspect graph coverage.
7. Run eval sets after changing retrieval logic.

## CLI Workflow

Ingest documents:

```powershell
python ingest.py add C:\path\to\documents
```

Build the evidence cards:

```powershell
python indexer.py
```

Build embeddings and entity/claim anchors:

```powershell
python embeddings.py build
python entity_claims.py build
```

Build canonical entities:

```powershell
python entity_canonicalizer.py build
```

Build graph relationships and community summaries:

```powershell
python knowledge_graph.py --build-summaries
```

Search:

```powershell
python searcher.py "Where does the project define power supply responsibilities?" --max-hits 12
```

Evaluate:

```powershell
python evaluator.py eval_sets/example_retrieval_eval.json --max-hits 12
```

## Retrieval Modes

The searcher adapts retrieval behavior by query type:

- `exact_lookup`
- `multi_hop`
- `comparison`
- `global_summary`
- `contradiction_check`
- `gap_analysis`
- `risk_analysis`
- `follow_up`
- `general`

Exact lookup emphasizes lexical, entity, and claim signals. Synthesis modes
use more graph, relationship, and community-summary context.

## Evaluation Format

Evaluation sets are JSON files with:

- question
- expected answer points
- expected evidence card
- expected document
- expected page
- required terms
- cross-document expectations
- forbidden answer terms

Reports are written to:

```text
projects/<project_id>/indexes/eval_reports/
```

Missed evidence is diagnosed as one of:

- `expected_document_never_entered_search_path`
- `no_matching_domain_visited`
- `domain_visited_but_cluster_missed`
- `cluster_visited_but_card_missed`
- `card_retrieved_but_page_mismatch`

## PostgreSQL Storage

PostgreSQL can be used as the primary storage for documents, pages, cards,
clusters, domains, relationships, search runs, and graph build runs.

```env
EVIDENCE_MESH_STORAGE=postgres
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/evidence_mesh
POSTGRES_CONNECT_TIMEOUT=5
EVIDENCE_MESH_PROJECT_ID=local
```

Initialize schema:

```powershell
python storage.py init-postgres
```

For local throwaway debugging only, use:

```env
EVIDENCE_MESH_STORAGE=json
```

## Data Layout

Runtime project data is written under:

```text
projects/<project_id>/
  documents/
  indexes/
  logs/
  uploads/
```

Important generated index files:

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

These runtime files can contain private documents or derived private content
and are ignored by Git.

## Privacy Boundary

Do not commit:

- `.env`
- uploaded documents
- generated project folders
- indexes
- search results
- eval reports from private projects
- logs
- caches
- PDFs, spreadsheets, CSVs, or zip exports

The repository should contain only source code, documentation, templates, and
small shareable eval fixtures.

## Verify

```powershell
python -m py_compile app.py ingest.py indexer.py embeddings.py entity_claims.py entity_canonicalizer.py community_summaries.py query_modes.py reranker.py knowledge_graph.py searcher.py evaluator.py
```

## License

MIT License. See the repository-level `LICENSE` file.

## Disclaimer

Evidence Mesh is a research and developer tool. Always verify important legal,
financial, engineering, or contractual conclusions against the original source
documents and expert human review.
