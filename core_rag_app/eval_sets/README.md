# Evaluation Sets

Evaluation sets are small, real quality checks for Evidence Mesh.

Each case should represent a question a user genuinely cares about, with the
evidence that a correct retrieval should find.

Use them after changing prompts, clustering, relationships, caching, or search
logic:

```powershell
python evaluator.py eval_sets/example_retrieval_eval.json --dry-run
```

For normal quality checks, omit `--dry-run` so the configured search model is used:

```powershell
python evaluator.py eval_sets/my_project_eval.json --max-hits 12
```

The runner writes a timestamped report under the active project at
`indexes/eval_reports/`.

For missed evidence, reports include `failure_diagnosis` with the most likely
retrieval break point:

- `no_matching_domain_visited`
- `domain_visited_but_cluster_missed`
- `cluster_visited_but_card_missed`
- `card_retrieved_but_page_mismatch`
- `expected_document_never_entered_search_path`
