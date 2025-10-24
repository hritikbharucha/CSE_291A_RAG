-- Table (as before)
CREATE TABLE IF NOT EXISTS docs (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  doc          TEXT    NOT NULL,
  meta         TEXT,
  access_freq  INTEGER NOT NULL DEFAULT 0,
  access_dt    TEXT,              -- UTC ISO-8601, e.g. 2025-10-23T14:02:00Z
  created_dt   TEXT    NOT NULL DEFAULT (DATETIME('now'))
);

CREATE INDEX IF NOT EXISTS idx_docs_sorted
ON docs(access_dt DESC, access_freq DESC, id ASC);
