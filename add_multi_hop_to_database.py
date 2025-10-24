import os
import json
import rag.rag as rag
from sentence_transformers import SentenceTransformer
import pandas as pd

if __name__ == '__main__':
    # adding Global News Dataset
    data_dir = "./data/MultiHopDataset"
    data_json = os.path.join(data_dir, "corpus.json")

    print("Loading dataset...")

    with open(data_json, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df = df.dropna(subset=["body"])

    articles = df["body"].tolist()
    meta_list = df.drop(columns=["body"]).to_dict(orient="records")

    articles = articles[:5000]
    meta_list = meta_list[:5000]

    my_rag = rag.RAG(
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
        rag_searcher = rag.FAISSRAGSearcher(384),
        dimension = 384,
        cache_size = 1000,
        db_dir = data_dir,
        db_name = "docs",
        new_db=True
    )

    print("Adding articles to database...")
    my_rag.add(articles, meta_list)
    my_rag.save()
    print("Articles added to database...")

