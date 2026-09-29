"""SQLite DDL for the observability ledger.

Design rules:
  * model_calls / context_segments / tool_calls are append-only facts.
  * Message *content* is never stored; only sizes, hashes and privacy-preserving
    fingerprints (hashed distinctive tokens) used for duplication/usage analysis.
  * budget_state is the only mutable table besides task/session terminal fields.
"""

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS sessions(
  session_id        TEXT PRIMARY KEY,
  parent_session_id TEXT,
  budget_root_id    TEXT NOT NULL,
  tenant            TEXT,
  root_task         TEXT,
  started_at        REAL NOT NULL,
  ended_at          REAL,
  terminal_status   TEXT,          -- completed | failed | budget_exhausted | abandoned
  checkpoint        TEXT           -- JSON resumable checkpoint (hard stop / wrap-up)
);
CREATE INDEX IF NOT EXISTS ix_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS ix_sessions_root ON sessions(budget_root_id);

CREATE TABLE IF NOT EXISTS budget_state(
  root_id              TEXT PRIMARY KEY,
  budget_limit         INTEGER NOT NULL,
  finalization_reserve INTEGER NOT NULL,
  basis                TEXT NOT NULL,        -- billed | fresh
  used                 INTEGER NOT NULL DEFAULT 0,
  reserved             INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS budget_events(
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  root_id      TEXT NOT NULL,
  session_id   TEXT,
  level        TEXT NOT NULL,                -- warning | high | critical | hard_stop | overshoot
  threshold    REAL,
  tokens_at    INTEGER NOT NULL,
  action_taken TEXT,
  ts           REAL NOT NULL,
  UNIQUE(root_id, level)
);

-- Reservations made through the HTTP API by external systems. Persisted so a server restart cannot leak
-- reserved tokens: expired rows are swept and their amount is released.
CREATE TABLE IF NOT EXISTS budget_reservations(
  reservation_id TEXT PRIMARY KEY,
  root_id        TEXT NOT NULL,
  session_id     TEXT NOT NULL,
  amount         INTEGER NOT NULL,
  est_input      INTEGER NOT NULL,
  granted_output INTEGER NOT NULL,
  finalization   INTEGER NOT NULL DEFAULT 0,
  expires_at     REAL NOT NULL
);

-- Pull-collector bookkeeping: skip transcript files that have not changed since the last pass.
CREATE TABLE IF NOT EXISTS collector_files(
  path       TEXT PRIMARY KEY,
  size       INTEGER NOT NULL,
  mtime      REAL NOT NULL,
  session_id TEXT,
  synced_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks(
  task_id         TEXT PRIMARY KEY,
  session_id      TEXT NOT NULL,
  task_type       TEXT,
  complexity_est  TEXT,                      -- simple | medium | complex | large
  outcome         TEXT,                      -- success | failure | partial | budget_exhausted | abandoned
  acceptance_ref  TEXT,
  started_at      REAL,
  ended_at        REAL,
  meta            TEXT
);
CREATE INDEX IF NOT EXISTS ix_tasks_session ON tasks(session_id);
CREATE INDEX IF NOT EXISTS ix_tasks_type ON tasks(task_type);

CREATE TABLE IF NOT EXISTS routing_decisions(
  decision_id         TEXT PRIMARY KEY,
  task_id             TEXT,
  session_id          TEXT,
  candidates          TEXT,                  -- JSON list of model ids considered
  chosen              TEXT NOT NULL,
  reason_code         TEXT,
  complexity_features TEXT,                  -- JSON
  router_version      TEXT,
  escalated_from      TEXT,                  -- model id this decision escalated away from
  ts                  REAL
);
CREATE INDEX IF NOT EXISTS ix_routing_task ON routing_decisions(task_id);

CREATE TABLE IF NOT EXISTS model_calls(
  seq                 INTEGER PRIMARY KEY AUTOINCREMENT,
  call_id             TEXT UNIQUE NOT NULL,
  session_id          TEXT NOT NULL,
  task_id             TEXT,
  agent_id            TEXT,
  agent_version       TEXT,
  prompt_version      TEXT,
  iteration           INTEGER,
  parent_call_id      TEXT,
  retry_of_call_id    TEXT,
  attempt_no          INTEGER NOT NULL DEFAULT 1,
  model_id            TEXT NOT NULL,
  provider            TEXT,
  routing_decision_id TEXT,
  input_fresh         INTEGER NOT NULL DEFAULT 0,
  cache_read          INTEGER NOT NULL DEFAULT 0,
  cache_write         INTEGER NOT NULL DEFAULT 0,
  output_tokens       INTEGER NOT NULL DEFAULT 0,
  reasoning_tokens    INTEGER NOT NULL DEFAULT 0,   -- SUBSET of output_tokens
  est_input_tokens    INTEGER,
  est_error           INTEGER,                      -- provider input - sum(segment tokens)
  cost_in             REAL,
  cost_out            REAL,
  price_table_version TEXT,
  latency_ms          REAL,
  ttft_ms             REAL,
  stop_reason         TEXT,
  status              TEXT NOT NULL,                -- success | failure | blocked_budget
  error_class         TEXT,
  budget_used_before  INTEGER,
  budget_used_after   INTEGER,
  tool_call_count     INTEGER NOT NULL DEFAULT 0,
  finalization        INTEGER NOT NULL DEFAULT 0,
  output_fingerprint  TEXT,                         -- JSON list of hashed distinctive tokens
  ts                  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_calls_session ON model_calls(session_id, seq);
CREATE INDEX IF NOT EXISTS ix_calls_task ON model_calls(task_id, seq);
CREATE INDEX IF NOT EXISTS ix_calls_model ON model_calls(model_id);

CREATE TABLE IF NOT EXISTS context_segments(
  segment_id      TEXT PRIMARY KEY,                 -- <call_id>:<position>
  call_id         TEXT NOT NULL,
  session_id      TEXT NOT NULL,
  kind            TEXT NOT NULL,   -- system|user|history|retrieval|tool_result|agent_state|tool_schema
  tokens          INTEGER NOT NULL,
  content_hash    TEXT NOT NULL,
  source_ref      TEXT,                             -- e.g. tool_call_id, doc id
  origin_call_id  TEXT,                             -- call in which this content first entered context
  position        INTEGER NOT NULL,
  age_iterations  INTEGER NOT NULL DEFAULT 0,
  was_truncated   INTEGER NOT NULL DEFAULT 0,
  fingerprint     TEXT
);
CREATE INDEX IF NOT EXISTS ix_seg_call ON context_segments(call_id, position);
CREATE INDEX IF NOT EXISTS ix_seg_hash ON context_segments(content_hash);
CREATE INDEX IF NOT EXISTS ix_seg_source ON context_segments(source_ref);

CREATE TABLE IF NOT EXISTS tool_calls(
  tool_call_id      TEXT PRIMARY KEY,
  call_id           TEXT NOT NULL,                  -- model call that issued it
  session_id        TEXT NOT NULL,
  tool_name         TEXT NOT NULL,
  args_hash         TEXT,
  status            TEXT NOT NULL,                  -- ok | error | memo_hit
  error_class       TEXT,
  retry_of          TEXT,
  latency_ms        REAL,
  raw_output_tokens INTEGER NOT NULL DEFAULT 0,
  injected_tokens   INTEGER NOT NULL DEFAULT 0,
  was_truncated     INTEGER NOT NULL DEFAULT 0,
  result_hash       TEXT,
  later_referenced  INTEGER                          -- NULL = not assessable; filled by waste analysis
);
CREATE INDEX IF NOT EXISTS ix_tool_session ON tool_calls(session_id);
CREATE INDEX IF NOT EXISTS ix_tool_call ON tool_calls(call_id);

CREATE TABLE IF NOT EXISTS quality_evals(
  eval_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id        TEXT NOT NULL,
  session_id     TEXT,
  method         TEXT NOT NULL,    -- test | assertion | human | user_signal | judge
  score          REAL NOT NULL,    -- 0..1
  rubric_version TEXT,
  judge_model    TEXT,
  confidence     REAL,
  meta           TEXT,
  ts             REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_quality_task ON quality_evals(task_id);
"""
