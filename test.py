import argparse
import os
import rag.rag as rag
from sentence_transformers import SentenceTransformer
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_file', type=str, default="mock_requests.jsonl")
    args = parser.parse_args()

    my_rag = rag.RAG(
        embedding_model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
        rag_searcher=rag.FAISSRAGSearcher(384),
        dimension=384,
        cache_size=1000,
        db_dir="./data",
        db_name="docs",
    )

    df = pd.read_json(args.query_file, lines=True)
    top1_acc = 0
    top5_acc = 0
    for index, row in df.iterrows():
        query = row["question"]
        gt_idx = row["id"] - 1

        top1_rslts = my_rag.retrieve([query], 1)
        top5_rslts = my_rag.retrieve([query], 5)
        top1_acc += 1 if gt_idx in top1_rslts.keys() else 0
        top5_acc += 1 if gt_idx in top5_rslts.keys() else 0

        # print(f"gt_id: {gt_idx}, query: {query}, top1_rslt: {top1_rslts.keys()}, top5_rslts: {top5_rslts.keys()}")

    print(f"Top-1 acc: {top1_acc/len(df)}, Top-5 acc: {top5_acc/len(df)}")

