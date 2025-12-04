import os
import argparse
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
    parser.add_argument("--vector_search_provider", type=str, default="hybrid",
                       choices=["faiss", "opensearch", "hybrid"],
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
    parser.add_argument("--sparse_type", type=str, default="bm25",
                       choices=["tfidf", "bm25"],
                       help="Sparse backend for hybrid search")
    parser.add_argument("--hybrid_alpha", type=float, default=0.5,
                       help="Weight for sparse scores when using hybrid search")
    args = parser.parse_args()
    
    # Get AWS region
    aws_region = get_aws_region(args.aws_region)
    
    # Resolve model names for Bedrock
    if args.embedding_provider == "bedrock":
        embedding_model_name = get_bedrock_embedding_model(args.embedding_model)
    else:
        embedding_model_name = args.embedding_model
    
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

    articles = articles[:args.num_articles]
    article_ids = article_ids[:args.num_articles]
    meta_list = meta_list[:args.num_articles]

    # create embedding provider first to get actual dimension
    embedding_provider = create_embedding_provider(
        provider_type=args.embedding_provider,
        model_name=embedding_model_name,
        dimension=None,  # let provider determine dimension
        region_name=aws_region
    )
    
    # use the actual dimension from the provider for vector search
    actual_dimension = embedding_provider.dimension
    
    vector_search_provider = create_vector_search_provider(
        provider_type=args.vector_search_provider,
        embedding_provider=embedding_provider,
        dimension=actual_dimension,
        endpoint=args.opensearch_endpoint,
        region_name=aws_region,
        sparse_type=args.sparse_type,
        alpha=args.hybrid_alpha
    )

    my_rag = rag.RAG(
        rag_searcher=vector_search_provider,
        cache_size=0, # don't need cache size in the init
        db_dir=args.db_dir,
        db_name=args.db_name,
        new_db=True
    )
    print("Adding articles to database...")
    my_rag.add(articles, article_ids, meta_list)
    my_rag.save() # save to index
    print("Articles added to database...")

