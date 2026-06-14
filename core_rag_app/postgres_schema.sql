CREATE TABLE IF NOT EXISTS documents (
  project_id TEXT NOT NULL,
  document_id TEXT NOT NULL,
  original_name TEXT NOT NULL,
  stored_path TEXT,
  source_path TEXT,
  extension TEXT,
  mime_type TEXT,
  size_bytes BIGINT,
  sha256 TEXT,
  ingested_at TIMESTAMPTZ,
  ingest_strategy TEXT,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, document_id)
);

CREATE TABLE IF NOT EXISTS pages (
  project_id TEXT NOT NULL,
  document_id TEXT NOT NULL,
  page_no INTEGER NOT NULL,
  document_name TEXT,
  card_count INTEGER NOT NULL DEFAULT 0,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, document_id, page_no)
);

CREATE TABLE IF NOT EXISTS cards (
  project_id TEXT NOT NULL,
  card_id TEXT NOT NULL,
  document_id TEXT NOT NULL,
  document_name TEXT,
  page_no INTEGER,
  card_name TEXT NOT NULL,
  card_description TEXT,
  card_source TEXT,
  tags TEXT[] NOT NULL DEFAULT '{}',
  schema_version TEXT,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, card_id)
);

CREATE TABLE IF NOT EXISTS clusters (
  project_id TEXT NOT NULL,
  cluster_id TEXT NOT NULL,
  document_id TEXT NOT NULL,
  document_name TEXT,
  cluster_name TEXT NOT NULL,
  cluster_description TEXT,
  card_ids TEXT[] NOT NULL DEFAULT '{}',
  schema_version TEXT,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS domains (
  project_id TEXT NOT NULL,
  domain_id TEXT NOT NULL,
  document_id TEXT NOT NULL,
  document_name TEXT,
  domain_name TEXT NOT NULL,
  domain_description TEXT,
  cluster_ids TEXT[] NOT NULL DEFAULT '{}',
  schema_version TEXT,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, domain_id)
);

CREATE TABLE IF NOT EXISTS relationships (
  project_id TEXT NOT NULL,
  relationship_id TEXT NOT NULL,
  main_domain_id TEXT NOT NULL,
  related_domain_id TEXT NOT NULL,
  relationship_type TEXT,
  document_scope TEXT,
  confidence_score DOUBLE PRECISION,
  evidence_strength DOUBLE PRECISION,
  source_coverage DOUBLE PRECISION,
  generation_method TEXT,
  schema_version TEXT,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, relationship_id)
);

CREATE TABLE IF NOT EXISTS relationship_pair_checks (
  project_id TEXT NOT NULL,
  pair_key TEXT NOT NULL,
  main_domain TEXT NOT NULL,
  related_domain TEXT NOT NULL,
  status TEXT NOT NULL,
  relationship_ids TEXT[] NOT NULL DEFAULT '{}',
  error TEXT,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, pair_key)
);

CREATE TABLE IF NOT EXISTS search_runs (
  project_id TEXT NOT NULL,
  search_run_id TEXT NOT NULL,
  query TEXT NOT NULL,
  hit_count INTEGER NOT NULL DEFAULT 0,
  schema_version TEXT,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, search_run_id)
);

CREATE TABLE IF NOT EXISTS graph_build_runs (
  project_id TEXT NOT NULL,
  graph_build_run_id TEXT NOT NULL,
  status TEXT NOT NULL,
  cluster_count INTEGER NOT NULL DEFAULT 0,
  domain_count INTEGER NOT NULL DEFAULT 0,
  relationship_count INTEGER NOT NULL DEFAULT 0,
  audit_status TEXT,
  audit_issue_count INTEGER NOT NULL DEFAULT 0,
  schema_version TEXT,
  payload JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (project_id, graph_build_run_id)
);

CREATE TABLE IF NOT EXISTS project_progress (
  project_id TEXT PRIMARY KEY,
  stage TEXT NOT NULL,
  message TEXT,
  payload JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_graph_state (
  project_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  cluster_count INTEGER NOT NULL DEFAULT 0,
  domain_count INTEGER NOT NULL DEFAULT 0,
  relationship_count INTEGER NOT NULL DEFAULT 0,
  relationship_checks_done INTEGER NOT NULL DEFAULT 0,
  relationship_checks_total INTEGER NOT NULL DEFAULT 0,
  failed_relationship_check_count INTEGER NOT NULL DEFAULT 0,
  schema_version TEXT,
  payload JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS graph_audits (
  project_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  issue_count INTEGER NOT NULL DEFAULT 0,
  schema_version TEXT,
  payload JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_sha256 ON documents (sha256);
CREATE INDEX IF NOT EXISTS idx_pages_document ON pages (project_id, document_id);
CREATE INDEX IF NOT EXISTS idx_cards_document_page ON cards (project_id, document_id, page_no);
CREATE INDEX IF NOT EXISTS idx_cards_name ON cards USING gin (to_tsvector('simple', card_name || ' ' || COALESCE(card_description, '')));
CREATE INDEX IF NOT EXISTS idx_clusters_document ON clusters (project_id, document_id);
CREATE INDEX IF NOT EXISTS idx_domains_document ON domains (project_id, document_id);
CREATE INDEX IF NOT EXISTS idx_relationships_main ON relationships (project_id, main_domain_id);
CREATE INDEX IF NOT EXISTS idx_relationships_related ON relationships (project_id, related_domain_id);
CREATE INDEX IF NOT EXISTS idx_relationships_quality ON relationships (project_id, confidence_score, evidence_strength, source_coverage);
CREATE INDEX IF NOT EXISTS idx_relationship_pair_checks_status ON relationship_pair_checks (project_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_search_runs_query ON search_runs USING gin (to_tsvector('simple', query));
CREATE INDEX IF NOT EXISTS idx_graph_build_runs_status ON graph_build_runs (project_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_project_progress_stage ON project_progress (stage, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_graph_state_status ON knowledge_graph_state (status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_graph_audits_status ON graph_audits (status, updated_at DESC);
