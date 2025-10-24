import os
import rag.rag as rag
from sentence_transformers import SentenceTransformer
import pandas as pd

if __name__ == '__main__':
    # adding Global News Dataset
    data_dir = "./data/GlobalNewsDataset"
    data_csv =  "data.csv"  # Kaggle Global News Dataset
    data_csv = os.path.join(data_dir, data_csv)

    print("Loading dataset...")
    df = pd.read_csv(data_csv)
    df = df.dropna(subset=["full_content"])

    articles = df["full_content"].tolist()
    article_ids = df["article_id"].tolist()
    article_ids = [f"global_news_ds_{article_id}" for article_id in article_ids]
    meta_list = df.drop(columns=["content", "full_content", "article_id", "source_id"]).to_dict(orient="records")

    articles = articles[:5000]
    meta_list = meta_list[:5000]

    my_rag = rag.RAG(
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
        rag_searcher = rag.FAISSRAGSearcher(384),
        dimension = 384,
        cache_size = 0, # don't need cache size in the init
        db_dir = data_dir,
        db_name = "docs",
        new_db=True
    )
    print("Adding articles to database...")
    my_rag.add(articles, article_ids, meta_list)
    my_rag.save() # save to index
    print("Articles added to database...")

