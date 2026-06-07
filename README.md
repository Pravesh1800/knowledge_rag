# PDF Vision RAG

PDF Vision RAG is a local FastAPI web app for uploading mixed tender/project
documents, indexing them page by page with vision-capable LLMs, building a
relationship map, searching the indexed corpus, and generating specialist bid
documents.

The app supports:

- PDF and spreadsheet ingestion
- page-level topic extraction
- relationship map generation across topics, communities, and biomes
- project chat/search
- generated reports for legal assessment, commercial strategy, financial bonds,
  pre-bid queries, and pre-qualification requirements

## 1. Requirements

Use Python 3.12 or newer.

Install Git if you are cloning the repository:

```powershell
git clone https://github.com/Pravesh1800/knowledge_rag.git
cd knowledge_rag
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install Python libraries:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Environment Setup

Create a `.env` file in the project root:

```powershell
New-Item .env -ItemType File
```

Add your OpenRouter configuration:

```text
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_SITE_URL=http://localhost:8020
OPENROUTER_APP_NAME=PDF Vision RAG

OPENROUTER_MODEL=google/gemini-3.1-flash-lite
OPENROUTER_SEARCH_MODEL=deepseek/deepseek-v4-flash
OPENROUTER_MAP_MODEL=google/gemini-3.1-flash-lite

OPENROUTER_LEGAL_AGENT_MODEL=deepseek/deepseek-v4.1
OPENROUTER_COMMERCIAL_AGENT_MODEL=deepseek/deepseek-v4.1
OPENROUTER_FINANCIAL_AGENT_MODEL=deepseek/deepseek-v4.1
OPENROUTER_PREBID_AGENT_MODEL=deepseek/deepseek-v4.1
OPENROUTER_PREQUAL_AGENT_MODEL=deepseek/deepseek-v4.1
```

You can change the model names if your OpenRouter account uses different model
IDs. The `.env` file is ignored by Git and should not be committed.

## 3. Run the Web App

Start the local server:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8020
```

Open the app:

```text
http://localhost:8020
```

## 4. Basic Workflow

1. Create or open a project in the web UI.
2. Upload all required documents.
3. Confirm the document list is complete.
4. Click `Start index generation`.
5. Wait for indexing and relationship mapping to complete.
6. Use project chat or run generated-doc agents.

Generated reports are saved under:

```text
projects/<project_id>/reports/
```

Indexes and relationship maps are saved under:

```text
projects/<project_id>/indexes/
```

These folders are ignored by Git because they can contain private documents,
large outputs, API-derived data, and generated reports.

## 5. Generated Documents

The UI can generate:

- Pre-Bid Queries
- Pre-Qualification Requirements
- Commercial Drivers and Strategy to WIN
- Financial Bonds
- Legal Assessment

Each report has progress tracking in the UI. Completed reports remain visible
even if a later rerun fails because of an API/key limit.

Several reports also include `Download Excel` buttons in the UI.

## 6. Optional Public Link With ngrok

If ngrok is installed and authenticated, expose the local app:

```powershell
ngrok http 8020
```

Use the HTTPS forwarding URL shown by ngrok.

Keep the FastAPI app running while using the ngrok URL.

## 7. CLI Commands

Most work should be done through the UI, but the pipeline scripts can also be
run directly.

Ingest a file or folder:

```powershell
python ingest.py add "C:\path\to\file-or-folder"
```

Build the topic index:

```powershell
python indexer.py --reset
```

Build the relationship map:

```powershell
python relationship_map.py
```

Search the indexed corpus:

```powershell
python searcher.py "commercial terms and risks"
```

Generate reports from the command line:

```powershell
python legal_assessment.py projects\<project_id>
python commercial_strategy.py projects\<project_id>
python financial_bonds.py projects\<project_id>
python prebid_queries.py projects\<project_id>
python prequalification_requirements.py projects\<project_id>
```

## 8. Smoke Checks

Check Python syntax:

```powershell
python -m py_compile app.py indexer.py ingest.py relationship_map.py searcher.py legal_assessment.py commercial_strategy.py financial_bonds.py prebid_queries.py prequalification_requirements.py
```

Check the running API:

```powershell
Invoke-RestMethod http://localhost:8020/api/projects
```

## 9. Notes

- Do not commit `.env`, uploaded documents, project indexes, reports, logs, or
  generated Excel/CSV/PDF files.
- Large indexing and relationship-map jobs can take time and consume API quota.
- If a generated-doc agent fails with an OpenRouter daily/key limit error,
  update the key or quota and rerun that report.
