import sqlite3, json, datetime as dt
import sys
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional
import tqdm

ISO = "%Y-%m-%dT%H:%M:%SZ"

def utc_now_iso() -> str:
    return dt.datetime.utcnow().strftime(ISO)

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
            CREATE TABLE IF NOT EXISTS docs (
              id           INTEGER PRIMARY KEY AUTOINCREMENT,
              doc          TEXT NOT NULL,
              meta         TEXT,
              access_freq  INTEGER NOT NULL DEFAULT 0,
              access_dt    TEXT,
              created_dt   TEXT NOT NULL DEFAULT (DATETIME('now'))
            );
            -- Composite index aligned with our desired ordering
            CREATE INDEX IF NOT EXISTS idx_docs_sorted
            ON docs(access_dt DESC, access_freq DESC, id ASC);
            """)

    def insert_docs(self, records: List[Dict[str, Any]])-> Dict[int, Dict[str, Any]]:
        new_rows = {}
        with self.conn:
            for row in tqdm.tqdm(records):
                cur = self.conn.execute(
                    """
                    INSERT INTO docs (doc, meta, access_freq, access_dt)
                    VALUES (?, ?, 0, NULL)
                    """,
                    (row["doc"], json.dumps(row.get("meta"), ensure_ascii=False))
                )
                # ids.append(cur.lastrowid)
                new_rows[cur.lastrowid] = {"doc": row["doc"], "meta": row.get("meta", {})}
        return new_rows

    def update_docs(self, records: List[Dict[str, Any]]):
        rows = [(r["id"], r["doc"], json.dumps(r.get("meta"), ensure_ascii=False))
                for r in records]
        with self.conn:
            self.conn.executemany("""
                                  INSERT INTO docs (id, doc, meta, access_freq, access_dt) 
                                  VALUES (?, ?, 0, NULL) 
                                  ON CONFLICT(id) DO 
                                    UPDATE SET 
                                        doc = excluded.doc, 
                                        meta = excluded.meta, 
                                        access_freq = docs.access_freq + 1, 
                                        access_dt = DATETIME('now'); 
                                  """, rows)

    def fetch_docs(self, ids: Iterable[int], touch: bool = True) -> Dict[int, Dict[str, Any]]:
        ids = list({int(i) for i in ids})
        if not ids:
            return {}
        q = ",".join("?" for _ in ids)
        with self.conn:
            rows = self.conn.execute(
                f"SELECT id, doc, meta, access_freq, access_dt FROM docs WHERE id IN ({q})",
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
            for (i, d, m, f, a) in rows:
                out[i] = {"doc": d, "meta": json.loads(m) if m else None,
                          "access_freq": f + 1, "access_datetime": now}
        else:
            for (i, d, m, f, a) in rows:
                out[i] = {"doc": d, "meta": json.loads(m) if m else None,
                          "access_freq": f, "access_datetime": a}
        return out

    def top_hot(self, limit: int = 20) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT id, doc, meta, access_freq, access_dt
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
