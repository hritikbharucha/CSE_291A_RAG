import sqlite3, json, datetime as dt
import sys
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional
import tqdm
import hashlib
import re

ISO = "%Y-%m-%dT%H:%M:%SZ"

def utc_now_iso() -> str:
    return dt.datetime.utcnow().strftime(ISO)

def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()

def compute_doc_hash(text: str) -> str:
    norm = normalize_text(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


# Splitting sentences
def split_into_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

def chunk_sentences(text: str, max_sentences: int = 5):
    sentences = split_into_sentences(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i+max_sentences])
        chunks.append(chunk)
    return chunks


class DocStore:
    def __init__(self, db_path: str = "data/docs.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")

    def init_schema(self):
        with self.conn:
            self.conn.executescript("""
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
            --    ON docs(article_id, doc_hash);
            
            CREATE INDEX IF NOT EXISTS idx_docs_article_id
                ON docs(article_id);
            
            CREATE INDEX IF NOT EXISTS idx_docs_sorted
                ON docs(access_dt DESC, access_freq DESC, id ASC);
            """)

    def insert_docs(self, records: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        article_ids, articles, metas = zip(*(
            (r["article_id"], r["doc"], r["meta"]) for r in records
        ))
        # insert articles table
        self.insert_article(article_ids, articles)
        rows = {}
        # for each article, we need to insert chunks
        for idx, article in tqdm.tqdm(enumerate(articles), total=len(articles), desc="Inserting articles into docs database"):
            chunks = chunk_sentences(article)
            article_id = article_ids[idx]
            meta = metas[idx]
            new_rows = self.insert_chunks(chunks, meta, article_id)
            rows = {**rows, **new_rows}
        return rows

    def insert_article(self, article_ids: List[str], contents: List[str]):
        with self.conn:
            self.conn.executemany(
            """
                INSERT INTO articles (article_id, content) VALUES (?, ?)
                ON CONFLICT (article_id) DO NOTHING;
                """,
                zip(article_ids, contents)
            )

    def fetch_article(self, article_id: str) -> str:
        with self.conn:
            cursor = self.conn.execute(
                "SELECT content FROM articles WHERE article_id = ?",
                (article_id,)
            )
            row = cursor.fetchone()
            return row[0]

    # insert chunk by 1 article
    def insert_chunks(self, doc_chunks: list[str], meta, article_id: str)-> Dict[int, Dict[str, Any]]:
        new_rows = {}
        article_title = meta["title"]
        article_description = f'a description: {meta["description"]}' if "description" in meta else "no description"
        format = "Quoted from {article_title}, which contains {article_description}. {chunk}"
        with self.conn:
            self.conn.execute("PRAGMA foreign_keys = ON;")
            for doc_chunk in doc_chunks:
                doc_chunk = format.format(article_title=article_title, article_description=article_description, chunk=doc_chunk)
                cur = self.conn.execute(
                    """
                    INSERT INTO docs (doc, meta, article_id, doc_hash, access_freq, access_dt)
                    VALUES (?, ?, ?, ?, 0, NULL)
                    ON CONFLICT (article_id, doc_hash) DO NOTHING;
                    """,
                    (
                        doc_chunk,
                        json.dumps(meta, ensure_ascii=False),
                        article_id,
                        compute_doc_hash(doc_chunk)
                    )
                )
                if cur.lastrowid is not None:
                    new_rows[cur.lastrowid] = {
                        "doc": doc_chunk,
                        "meta": meta,
                        "article_id": article_id
                    }
        return new_rows

    # Does not have any case for update so far
    # def update_docs(self, records: List[Dict[str, Any]]):
    #     rows = [(r["id"], r["doc"], json.dumps(r.get("meta"), ensure_ascii=False))
    #             for r in records]
    #     with self.conn:
    #         self.conn.executemany("""
    #                               INSERT INTO docs (id, doc, meta, access_freq, access_dt)
    #                               VALUES (?, ?, 0, NULL)
    #                               ON CONFLICT(id) DO
    #                                 UPDATE SET
    #                                     doc = excluded.doc,
    #                                     meta = excluded.meta,
    #                                     access_freq = docs.access_freq + 1,
    #                                     access_dt = DATETIME('now');
    #                               """, rows)

    def fetch_chunks(self, ids: Iterable[int], touch: bool = True) -> Dict[int, Dict[str, Any]]:
        ids = list({int(i) for i in ids})
        if not ids:
            return {}
        q = ",".join("?" for _ in ids)
        with self.conn:
            rows = self.conn.execute(
                f"SELECT id, doc, meta, article_id, access_freq, access_dt FROM docs WHERE id IN ({q})",
                ids
            ).fetchall()

        out: Dict[int, Dict[str, Any]] = {}
        if touch and rows:
            now = utc_now_iso()
            with self.conn:
                self.conn.executemany(
                    "UPDATE docs SET access_freq = access_freq + 1, access_dt = ? WHERE id = ?",
                    [(now, r[0]) for r in rows]
                )
            for (i, d, m, a_id, f, a) in rows:
                out[i] = {"doc": d, "meta": json.loads(m) if m else None, "article_id": a_id,
                          "access_freq": f + 1, "access_datetime": now}
        else:
            for (i, d, m, a_id, f, a) in rows:
                out[i] = {"doc": d, "meta": json.loads(m) if m else None, "article_id": a_id,
                          "access_freq": f, "access_datetime": a}
        return out

    def top_hot(self, limit: int = 20) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT id, doc, meta, access_freq, access_dt, article_id
            FROM docs
            ORDER BY (access_dt IS NULL) ASC,
                     access_dt DESC,
                     access_freq DESC,
                     id ASC LIMIT ?;
            """,
            (limit, )
        ).fetchall()

        return [
            {
                "id": int(r[0]),
                "doc": r[1],
                "meta": json.loads(r[2]) if r[2] else None,
                "access_freq": r[3],
                "access_datetime": r[4],
                "article_id": r[5]
            }
            for r in rows
        ]

    def vacuum(self):
        with self.conn:
            self.conn.execute("VACUUM;")
            self.conn.execute("ANALYZE;")

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    if len(sys.argv[1:]) > 0:
        db_name = sys.argv[1]
    else:
        db_name = "docs"

    store = DocStore(db_path=f"data/{db_name}.sqlite")
    store.init_schema()
    print(f"Initialized database at {store.db_path}")
