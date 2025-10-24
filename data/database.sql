-- create table for saving article (for testing)
CREATE TABLE IF NOT EXISTS articles (
    article_id   TEXT PRIMARY KEY,
    content    TEXT NOT NULL
);
-- create main table for RAG
CREATE TABLE IF NOT EXISTS docs (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  doc          TEXT NOT NULL,
  meta         TEXT,
  article_id   TEXT NOT NULL,
  doc_hash     TEXT NOT NULL,
  access_freq  INTEGER NOT NULL DEFAULT 0,
  access_dt    TEXT,
  created_dt   TEXT NOT NULL DEFAULT (DATETIME('now')),
  FOREIGN KEY (article_id) REFERENCES articles(article_id) ON DELETE CASCADE,
  UNIQUE(article_id, doc_hash)
);
-- Prevent duplicate chunks for the same article
-- CREATE UNIQUE INDEX IF NOT EXISTS ux_docs_article_hash
--     ON docs(article_id, doc_hash);

CREATE INDEX IF NOT EXISTS idx_docs_article_id
    ON docs(article_id);

CREATE INDEX IF NOT EXISTS idx_docs_sorted
    ON docs(access_dt DESC, access_freq DESC, id ASC);