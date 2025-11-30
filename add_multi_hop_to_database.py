import os
import json
import argparse
import numpy as np

import rag.rag as rag
from rag.provider_factory import create_embedding_provider, create_vector_search_provider
from rag.aws_config import get_aws_region, get_bedrock_embedding_model
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_provider", type=str, default="sentence_transformer",
                       choices=["sentence_transformer", "bedrock"],
                       help="Embedding provider type")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Embedding model name")
    parser.add_argument("--vector_search_provider", type=str, default="faiss",
                       choices=["faiss", "opensearch"],
                       help="Vector search provider type")
    parser.add_argument("--aws_region", type=str, default=None,
                       help="AWS region (defaults to AWS_REGION env var or us-east-1)")
    parser.add_argument("--opensearch_endpoint", type=str, default=None,
                       help="OpenSearch endpoint URL (required for opensearch provider)")
    parser.add_argument("--dimension", type=int, default=384,
                       help="Embedding dimension")
    parser.add_argument("--db_dir", type=str, default="./data",
                       help="RAG database directory")
    parser.add_argument("--db_name", type=str, default="docs",
                       help="RAG database name")
    parser.add_argument("--num_articles", type=int, default=5000,
                       help="Number of articles to add")
    args = parser.parse_args()
    
    # Get AWS region
    aws_region = get_aws_region(args.aws_region)
    
    # Resolve model names for Bedrock
    if args.embedding_provider == "bedrock":
        embedding_model_name = get_bedrock_embedding_model(args.embedding_model)
    else:
        embedding_model_name = args.embedding_model
    
    # adding MultiHop Dataset
    data_dir = "./data/MultiHopDataset"
    data_json = os.path.join(data_dir, "corpus.json")

    print("Loading dataset...")

    with open(data_json, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df = df.dropna(subset=["body"])

    articles = df["body"].tolist()
    meta_list = df.drop(columns=["body"]).to_dict(orient="records")

    articles = articles[:args.num_articles]
    article_ids = np.arange(len(articles)) #TODO: need to update when using different splitting
    article_ids = [f"multi_hop_ds_{article_id}" for article_id in article_ids]
    meta_list = meta_list[:args.num_articles]

    embedding_provider = create_embedding_provider(
        provider_type=args.embedding_provider,
        model_name=embedding_model_name,
        dimension=None,
        region_name=aws_region
    )
    
    actual_dimension = embedding_provider.dimension
    
    vector_search_provider = create_vector_search_provider(
        provider_type=args.vector_search_provider,
        dimension=actual_dimension,
        endpoint=args.opensearch_endpoint,
        region_name=aws_region
    )

    my_rag = rag.RAG(
        embedding_model=embedding_provider,
        rag_searcher=vector_search_provider,
        cache_size=0, # don't need cache size in the init
        db_dir=args.db_dir,
        db_name=args.db_name,
        new_db=True
    )

    print("Adding articles to database...")
    my_rag.add(articles, article_ids, meta_list)
    my_rag.save()
    print("Articles added to database...")

