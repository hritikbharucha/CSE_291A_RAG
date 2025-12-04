from typing import Optional
from .embedding_provider import (
    BaseEmbeddingProvider, 
    SentenceTransformerProvider, 
    BedrockEmbeddingProvider,
    create_embedding_provider as _create_embedding_provider
)
from .llm_provider import (
    BaseLLMProvider,
    HuggingFaceLLMProvider,
    BedrockLLMProvider,
    create_llm_provider as _create_llm_provider
)
from .searcher import RAGSearcher, FAISSRAGSearcher, OpenSearchProvider, HybridRAGSearcher


def create_embedding_provider(
    provider_type: str = "sentence_transformer",
    model_name: Optional[str] = None,
    dimension: Optional[int] = None,
    region_name: str = "us-east-1",
    **kwargs
) -> BaseEmbeddingProvider:
    if provider_type == "bedrock":
        kwargs["region_name"] = region_name
    
    provider = _create_embedding_provider(provider_type, model_name, **kwargs)
    
    if dimension is not None:
        if provider.dimension != dimension:
            import warnings
            warnings.warn(
                f"Provider dimension ({provider.dimension}) does not match "
                f"expected dimension ({dimension}). Using provider's dimension ({provider.dimension}).",
                UserWarning
            )
    else:
        dimension = provider.dimension
    
    return provider


def create_vector_search_provider(
    provider_type: str = "faiss",
    embedding_provider: Optional[BaseEmbeddingProvider] = None,
    dimension: Optional[int] = 384,
    endpoint: Optional[str] = None,
    index_name: str = "rag_index",
    region_name: str = "us-east-1",
    sparse_type: str = "tfidf",
    alpha: float = 0.5,
    **kwargs
) -> RAGSearcher:
    if embedding_provider is None and provider_type in {"faiss", "opensearch", "hybrid"}:
        raise ValueError("embedding_provider is required for dense or hybrid searchers.")

    if provider_type == "faiss":
        return FAISSRAGSearcher(embedding_provider=embedding_provider, dimension=dimension, alpha=alpha, **kwargs)
    elif provider_type == "opensearch":
        if endpoint is None:
            raise ValueError("endpoint is required for OpenSearch provider")
        return OpenSearchProvider(
            embedding_provider=embedding_provider,
            dimension=dimension,
            endpoint=endpoint,
            index_name=index_name,
            region_name=region_name,
            alpha=alpha,
            **kwargs
        )
    elif provider_type == "hybrid":
        return HybridRAGSearcher(
            embedding_provider=embedding_provider,
            dimension=dimension,
            sparse_type=sparse_type,
            alpha=alpha,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown vector search provider type: {provider_type}")


def create_llm_provider(
    provider_type: str = "huggingface",
    model_name: Optional[str] = None,
    region_name: str = "us-east-1",
    **kwargs
) -> BaseLLMProvider:
    if provider_type == "bedrock":
        kwargs["region_name"] = region_name
    
    return _create_llm_provider(provider_type, model_name, **kwargs)


def create_providers(
    embedding_provider_type: str = "sentence_transformer",
    embedding_model_name: Optional[str] = None,
    vector_search_provider_type: str = "faiss",
    llm_provider_type: str = "huggingface",
    llm_model_name: Optional[str] = None,
    dimension: int = 384,
    aws_region: str = "us-east-1",
    opensearch_endpoint: Optional[str] = None,
    opensearch_index_name: str = "rag_index",
    **kwargs
):
    embedding_provider = create_embedding_provider(
        provider_type=embedding_provider_type,
        model_name=embedding_model_name,
        dimension=dimension,
        region_name=aws_region,
        **kwargs.get("embedding_kwargs", {})
    )
    
    vector_search_provider = create_vector_search_provider(
        provider_type=vector_search_provider_type,
        embedding_provider=embedding_provider,
        dimension=dimension,
        endpoint=opensearch_endpoint,
        index_name=opensearch_index_name,
        region_name=aws_region,
        sparse_type=kwargs.get("sparse_type", "tfidf"),
        alpha=kwargs.get("alpha", 0.5),
        **kwargs.get("vector_search_kwargs", {})
    )
    
    llm_provider = create_llm_provider(
        provider_type=llm_provider_type,
        model_name=llm_model_name,
        region_name=aws_region,
        **kwargs.get("llm_kwargs", {})
    )
    
    return embedding_provider, vector_search_provider, llm_provider

