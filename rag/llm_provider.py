from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 120, 
                 temperature: float = 0.7, top_p: float = 0.95, **kwargs) -> str:
        """
        Generate text from a prompt.
        :param prompt: Input text prompt
        :param max_new_tokens: Maximum number of tokens to generate
        :param temperature: Sampling temperature
        :param top_p: Nucleus sampling parameter
        :return: Generated text string
        """
        pass


class HuggingFaceLLMProvider(BaseLLMProvider):
    def __init__(self, model_name: str, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        device_map = kwargs.pop("device_map", "auto")
        torch_dtype = kwargs.pop("torch_dtype", "auto")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            **kwargs
        )
        
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.pad_token_id,
        )
    
    def generate(self, prompt: str, max_new_tokens: int = 120, 
                 temperature: float = 0.7, top_p: float = 0.95, **kwargs) -> str:
        result = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            **kwargs
        )
        return result[0]["generated_text"]


class BedrockLLMProvider(BaseLLMProvider):
    def __init__(self, model_name: str, region_name: str = "us-east-1", **kwargs):
        import boto3
        self.model_name = model_name
        self.region_name = region_name
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region_name)
    
    def generate(self, prompt: str, max_new_tokens: int = 120, 
                 temperature: float = 0.7, top_p: float = 0.95, **kwargs) -> str:
        import json
        
        if "claude" in self.model_name.lower():
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        elif "llama" in self.model_name.lower():
            body = {
                "prompt": prompt,
                "max_gen_len": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        elif "mistral" in self.model_name.lower():
            body = {
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        elif "titan" in self.model_name.lower():
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_new_tokens,
                    "temperature": temperature,
                    "topP": top_p,
                }
            }
        else:
            body = {
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        
        response = self.bedrock_runtime.invoke_model(
            modelId=self.model_name,
            body=json.dumps(body).encode('utf-8')
        )
        
        response_body = json.loads(response['body'].read())
        
        if "claude" in self.model_name.lower():
            if "content" in response_body:
                return response_body["content"][0]["text"]
            else:
                return response_body.get("completion", "")
        elif "llama" in self.model_name.lower():
            return response_body.get("generation", "")
        elif "mistral" in self.model_name.lower():
            return response_body.get("outputs", [{}])[0].get("text", "")
        elif "titan" in self.model_name.lower():
            return response_body.get("results", [{}])[0].get("outputText", "")
        else:
            return (response_body.get("completion") or
                   response_body.get("generated_text") or 
                   response_body.get("text") or 
                   str(response_body))


def create_llm_provider(provider_type: str = "huggingface", 
                        model_name: str = None, 
                        **kwargs) -> BaseLLMProvider:
    if provider_type == "huggingface":
        if model_name is None:
            model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        return HuggingFaceLLMProvider(model_name, **kwargs)
    elif provider_type == "bedrock":
        if model_name is None:
            model_name = "anthropic.claude-3-sonnet-20240229-v1:0"
        return BedrockLLMProvider(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown LLM provider type: {provider_type}")


