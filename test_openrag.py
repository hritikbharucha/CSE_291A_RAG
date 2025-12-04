import argparse
import sys
import os
import json
import pandas as pd
import tqdm
import rag.rag as rag
from rag.provider_factory import create_embedding_provider, create_vector_search_provider
from rag.aws_config import get_aws_region, get_bedrock_embedding_model
from pathlib import Path
from typing import List, Dict, Any


def calculate_retrieval_metrics(retrieved_lists: List[List[str]], 
                                 gold_lists: List[List[str]], 
                                 max_k: int = 10) -> Dict[str, float]:
    """
    Calculate retrieval metrics similar to Open RAG Eval and MultiHop-RAG.
    
    Metrics computed:
    - Hits@k: Whether any gold document is in top-k retrieved
    - MRR@k: Mean Reciprocal Rank
    - MAP@k: Mean Average Precision
    - Precision@k: Precision at k
    - Recall@k: Recall at k
    """
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    mrr_list = []
    map_list = []
    precision_at_5_list = []
    recall_at_5_list = []
    
    for retrieved, gold in zip(retrieved_lists, gold_lists):
        if not gold:
            continue
            
        # normalize text for comparison
        gold_normalized = [g.replace(" ", "").replace("\n", "").lower() for g in gold]
        retrieved_normalized = [r.replace(" ", "").replace("\n", "").lower() for r in retrieved]
        
        # find first relevant rank
        first_relevant_rank = None
        relevant_count = 0
        precision_sum = 0
        
        for rank, retrieved_item in enumerate(retrieved_normalized[:max_k], start=1):
            is_relevant = any(gold_item in retrieved_item for gold_item in gold_normalized)
            
            if is_relevant:
                if first_relevant_rank is None:
                    first_relevant_rank = rank
                relevant_count += 1
                precision_sum += relevant_count / rank
                
                if rank <= 1:
                    hits_at_1 += 1
                if rank <= 5:
                    hits_at_5 += 1
                if rank <= 10:
                    hits_at_10 += 1
                break  # For Hits@k, we only count first hit
        
        # MRR
        mrr_list.append(1 / first_relevant_rank if first_relevant_rank else 0)
        
        # MAP
        if relevant_count > 0:
            map_list.append(precision_sum / min(len(gold), max_k))
        else:
            map_list.append(0)
        
        # Precision@5 and Recall@5
        relevant_in_top5 = sum(1 for r in retrieved_normalized[:5] 
                               if any(g in r for g in gold_normalized))
        precision_at_5_list.append(relevant_in_top5 / 5)
        recall_at_5_list.append(relevant_in_top5 / len(gold) if gold else 0)
    
    n = len(gold_lists)
    return {
        'Hits@1': hits_at_1 / n if n > 0 else 0,
        'Hits@5': hits_at_5 / n if n > 0 else 0,
        'Hits@10': hits_at_10 / n if n > 0 else 0,
        'MRR@10': sum(mrr_list) / n if n > 0 else 0,
        'MAP@10': sum(map_list) / n if n > 0 else 0,
        'Precision@5': sum(precision_at_5_list) / n if n > 0 else 0,
        'Recall@5': sum(recall_at_5_list) / n if n > 0 else 0,
    }


def run_openrag_eval(args):
    print("Open RAG Style Retrieval Evaluation")
    
    print(f"Loading RAG system from {args.db_dir}...")
    
    # Get AWS region
    aws_region = get_aws_region(args.aws_region)
    
    # Resolve model names for Bedrock
    if args.embedding_provider == "bedrock":
        embedding_model_name = get_bedrock_embedding_model(args.embedding_model)
    else:
        embedding_model_name = args.embedding_model
    
    # Create providers
    embedding_provider = create_embedding_provider(
        provider_type=args.embedding_provider,
        model_name=embedding_model_name,
        dimension=args.dimension,
        region_name=aws_region
    )
    
    vector_search_provider = create_vector_search_provider(
        provider_type=args.vector_search_provider,
        embedding_provider=embedding_provider,
        dimension=args.dimension,
        endpoint=args.opensearch_endpoint,
        region_name=aws_region,
        sparse_type=args.sparse_type,
        alpha=args.hybrid_alpha
    )
    
    my_rag = rag.RAG(
        rag_searcher=vector_search_provider,
        cache_size=args.cache_size,
        db_dir=args.db_dir,
        db_name=args.db_name,
    )
    
    print(f"Loading queries from {args.query_file}...")
    df = pd.read_json(args.query_file, lines=True)
    print(f"Loaded {len(df)} queries")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    retrieved_lists = []
    gold_lists = []
    results = []
    
    chunk_top1_acc = 0
    chunk_top5_acc = 0
    article_top1_acc = 0
    article_top5_acc = 0
    total = 0
    
    pbar = tqdm.tqdm(df.iterrows(), total=len(df), desc="Retrieving documents")
    
    for idx, row in pbar:
        query = str(row["question"])
        
        try:
            gt_chunk_id = int(row["id"])  # Database IDs are 1-indexed, no need to subtract 1
        except (KeyError, ValueError, TypeError):
            gt_chunk_id = None
        try:
            gt_article_id = str(row["article_id"])
        except (KeyError, ValueError, TypeError):
            gt_article_id = None
        try:
            gt_chunk_text = str(row["chunk"])
        except (KeyError, ValueError, TypeError):
            gt_chunk_text = ""
        
        retrieved = my_rag.retrieve([query], args.top_k)
        
        retrieved_texts = []
        retrieved_ids = []
        retrieved_article_ids = []
        
        for chunk_id, chunk_info in retrieved.items():
            retrieved_texts.append(chunk_info['doc'])
            retrieved_ids.append(chunk_id)
            retrieved_article_ids.append(chunk_info.get('article_id', ''))
        
        # For OpenRAG-style metrics, use chunk IDs as the gold signal.
        retrieved_lists.append([str(i) for i in retrieved_ids])
        gold_lists.append([str(gt_chunk_id)] if gt_chunk_id is not None else [])
        
        if gt_chunk_id is not None:
            chunk_top1_acc += 1 if gt_chunk_id in retrieved_ids[:1] else 0
            chunk_top5_acc += 1 if gt_chunk_id in retrieved_ids[:5] else 0
        
        if gt_article_id:
            article_top1_acc += 1 if gt_article_id in retrieved_article_ids[:1] else 0
            article_top5_acc += 1 if gt_article_id in retrieved_article_ids[:5] else 0
        
        total += 1
        
        pbar.set_description(
            f"Chunk Top1: {chunk_top1_acc/total:.4f}, Top5: {chunk_top5_acc/total:.4f} | "
            f"Article Top1: {article_top1_acc/total:.4f}, Top5: {article_top5_acc/total:.4f}"
        )
        
        results.append({
            "query_id": str(idx),
            "query": query,
            "retrieved_texts": retrieved_texts,
            "retrieved_ids": [str(i) for i in retrieved_ids],
            "ground_truth_text": gt_chunk_text,
            "ground_truth_id": str(gt_chunk_id) if gt_chunk_id is not None else "",
            "ground_truth_article_id": gt_article_id or "",
        })
    
    print("Calculating Retrieval Metrics...")
    
    metrics = calculate_retrieval_metrics(retrieved_lists, gold_lists, max_k=args.top_k)
    
    print("RETRIEVAL EVALUATION RESULTS")
    
    print("\n--- Chunk-Level Accuracy (ID Match) ---")
    print(f"Top-1 Accuracy: {chunk_top1_acc / total:.4f}")
    print(f"Top-5 Accuracy: {chunk_top5_acc / total:.4f}")
    
    print("\n--- Article-Level Accuracy (ID Match) ---")
    print(f"Top-1 Accuracy: {article_top1_acc / total:.4f}")
    print(f"Top-5 Accuracy: {article_top5_acc / total:.4f}")
    
    print("\n--- ID-Based Retrieval Metrics (Chunk ID) ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"Total queries evaluated: {total}")
    
    # Save results to files
    json_path = output_dir / "openrag_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {json_path}")
    
    # Save metrics summary
    metrics_summary = {
        "chunk_top1_accuracy": chunk_top1_acc / total,
        "chunk_top5_accuracy": chunk_top5_acc / total,
        "article_top1_accuracy": article_top1_acc / total,
        "article_top5_accuracy": article_top5_acc / total,
        **metrics,
        "total_queries": total,
    }
    
    metrics_path = output_dir / "openrag_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved metrics summary to {metrics_path}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query_file",
        type=str,
        default="mock_requests.jsonl",
        help="Path to query file (JSONL format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="openrag_output",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        default="./data",
        help="RAG database directory",
    )
    parser.add_argument(
        "--db_name",
        type=str,
        default="docs",
        help="RAG database name",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of documents to retrieve per query",
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=0,
        help="LRU cache size for RAG system (0 disables caching)",
    )
    parser.add_argument(
        "--embedding_provider",
        type=str,
        default="sentence_transformer",
        choices=["sentence_transformer", "bedrock"],
        help="Embedding provider type",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name",
    )
    parser.add_argument(
        "--vector_search_provider",
        type=str,
        default="faiss",
        choices=["faiss", "opensearch", "hybrid"],
        help="Vector search provider type",
    )
    parser.add_argument(
        "--aws_region",
        type=str,
        default=None,
        help="AWS region (defaults to AWS_REGION env var or us-east-1)",
    )
    parser.add_argument(
        "--opensearch_endpoint",
        type=str,
        default=None,
        help="OpenSearch endpoint URL (required for opensearch provider)",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=384,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--sparse_type",
        type=str,
        default="bm25",
        choices=["tfidf", "bm25"],
        help="Sparse backend for hybrid search",
    )
    parser.add_argument(
        "--hybrid_alpha",
        type=float,
        default=0.5,
        help="Weight for sparse scores when using hybrid search",
    )
    args = parser.parse_args()

    run_openrag_eval(args)
    