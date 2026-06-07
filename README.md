# PDF Vision RAG

Custom document indexing pipeline for mixed document types.

## Point 1: Ingest Any Document

Put documents into the project with:

```powershell
python ingest.py add "C:\path\to\your\file.pdf"
```

You can also ingest a whole folder:

```powershell
python ingest.py add "C:\path\to\folder"
```

Files are copied into `documents/originals`, and metadata is written to
`documents/manifest.json`.

## Point 2: Page-by-Page Topic Indexing

Create a `.env` file:

```powershell
Copy-Item .env.example .env
```

Then set your OpenRouter key in `.env`:

```text
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=google/gemini-3.1-flash-lite
OPENROUTER_SEARCH_MODEL=deepseek/deepseek-v4-flash
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Run a real OpenRouter index:

```powershell
python indexer.py --reset
```

Run a cheap local smoke test without Gemini:

```powershell
python indexer.py --dry-run --limit-pages 2 --reset
```

The index is written to `indexes/topic_index.json`.

For PDFs, every page is rendered into `indexes/pages/<document-id>/page_0001.png`
so the OpenRouter vision model can inspect visual content. Each topic entry stores:

- `topic_name`
- `topic_description`
- `document_name`
- `page_no`
- `content`
- `topic_source`: `text`, `image`, or `mixed`
- `tags`: includes `image` for visual/image-derived topics
- `image_descriptions`

## Topic Versioning

During real indexing, each extracted topic is compared against existing index
entries with OpenRouter. If the model decides the new topic has a similar name
and is similar to or an extension of existing content, the indexer stores it as
the next version:

```text
Commercial Overview
Commercial Overview_v2
Commercial Overview_v3
```

The suffix is assigned by code so repeated versions stay consistent.

## Relationship Map

After creating `indexes/topic_index.json`, build communities and biomes:

```powershell
python relationship_map.py
```

Cheap local smoke test:

```powershell
python relationship_map.py --dry-run
```

This writes `indexes/relationship_map.json` with:

- `communities`: grouped similar topics within each document
- `biomes`: grouped similar communities within each document
- `biome_relations`: every detected directed relation between biomes

Community and biome names are globally unique. If a name repeats, the script
uses version suffixes like `_v2`, `_v3`.

Each biome relation stores:

- `main_biome`
- `related_biome`
- `relation_type`
- `relation_description`
- `evidence`
- `community_links`: exact community-to-community connections inside that biome relation
- `topic_links`: exact topic-to-topic connections inside that biome relation
- `document_scope`: `same_document` or `cross_document`

## Recursive Tree Search

Search broad questions by walking the relationship tree:

```powershell
python searcher.py "commercial terms and risks"
```

Cheap local smoke test:

```powershell
python searcher.py "commercial terms and risks" --dry-run
```

The searcher walks:

```text
biomes -> communities -> topics -> content
```

If a branch does not produce useful topic hits, it backtracks and follows
related biome branches. A node is never visited twice in the same search, so the
search cannot loop on repeated relations.

Results are written to `indexes/search_results/`.

When a relevant topic is found, the searcher also checks topic-to-topic links
inside `biome_relations`. Helpful related topics are returned under each hit as
`related_topics`, including their relation type, relation description, page
reference, and content.
