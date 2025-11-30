import os
from typing import Optional, Dict


DEFAULT_REGION = "us-east-1"

BEDROCK_EMBEDDING_MODELS = {
    "titan-v1": "amazon.titan-embed-text-v1",
    "cohere-english": "cohere.embed-english-v3",
    "cohere-multilingual": "cohere.embed-multilingual-v3",
}

BEDROCK_LLM_MODELS = {
    # Claude models (claude-3-haiku and claude-3-opus require Marketplace subscription)
    "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    # Llama 3 models
    "llama3-8b": "meta.llama3-8b-instruct-v1:0",
    "llama3-70b": "meta.llama3-70b-instruct-v1:0",
    "llama3-1-70b": "meta.llama3-1-70b-instruct-v1:0",
    # Mistral models
    "mistral-7b": "mistral.mistral-7b-instruct-v0:2",
    "mistral-large": "mistral.mistral-large-2402-v1:0",
    # Titan models (note: titan-text has formatting issues, titan-express untested)
    "titan-text": "amazon.titan-text-lite-v1",
    "titan-express": "amazon.titan-text-express-v1",
}


def get_aws_region(region: Optional[str] = None) -> str:
    if region:
        return region
    return os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", DEFAULT_REGION))


def get_bedrock_embedding_model(model_name: str) -> str:
    if model_name.startswith("amazon.") or model_name.startswith("cohere."):
        return model_name
    
    if model_name in BEDROCK_EMBEDDING_MODELS:
        return BEDROCK_EMBEDDING_MODELS[model_name]
    
    return model_name


def get_bedrock_llm_model(model_name: str) -> str:
    if "." in model_name and ":" in model_name:
        return model_name
    
    if model_name in BEDROCK_LLM_MODELS:
        return BEDROCK_LLM_MODELS[model_name]
    
    return model_name


def validate_aws_credentials() -> bool:
    try:
        import boto3
        session = boto3.Session()
        credentials = session.get_credentials()
        return credentials is not None
    except ImportError:
        return False
    except Exception:
        return False


def get_opensearch_endpoint(endpoint: Optional[str] = None) -> Optional[str]:
    if endpoint:
        return endpoint
    return os.environ.get("OPENSEARCH_ENDPOINT", None)


def list_available_bedrock_models(region: Optional[str] = None) -> Dict[str, list]:
    try:
        import boto3
        region = get_aws_region(region)
        bedrock = boto3.client('bedrock', region_name=region)
        
        response = bedrock.list_foundation_models()
        models = response.get('modelSummaries', [])
        
        embedding_models = []
        llm_models = []
        
        for model in models:
            model_id = model.get('modelId', '')
            if 'embed' in model_id.lower() or 'titan-embed' in model_id.lower():
                embedding_models.append(model_id)
            else:
                llm_models.append(model_id)
        
        return {
            'embedding': sorted(embedding_models),
            'llm': sorted(llm_models)
        }
    except Exception as e:
        print(f"Error listing Bedrock models: {e}")
        return {'embedding': [], 'llm': []}

