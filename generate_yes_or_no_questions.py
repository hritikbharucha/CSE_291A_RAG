import argparse
import numpy as np
import pandas as pd
import json
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath('')))

import rag.rag as rag
from sentence_transformers import SentenceTransformer

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_question(text: str) -> str:
    matches = re.findall(r"<question>\s*(.*?)\s*</question>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        # return the first match
        return matches[-1].strip()

    cleaned = re.sub(r"<[^>]+>", "", text).strip()
    for line in cleaned.splitlines():
        s = line.strip()
        if s:
            return s
    return cleaned

def extract_answer(text: str) -> str:
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        # return the first match
        return matches[-1].strip()

    cleaned = re.sub(r"<[^>]+>", "", text).strip()
    for line in cleaned.splitlines():
        s = line.strip()
        if s:
            return s
    return cleaned

def truncate_article(article: str, max_chars: int = 4000) -> str:
    if len(article) <= max_chars:
        return article
    head = article[: max_chars // 2]
    tail = article[- max_chars // 2 :]
    return head + "\n...\n" + tail

def generate_retrieval_queries(num_queries: int = 100) -> list:
    base_queries = [
        "recent news about technology",
        "global events and politics",
        "economic developments",
        "scientific discoveries",
        "environmental issues",
        "health and medicine",
        "international relations",
        "business and finance",
        "social issues",
        "cultural events",
        "sports news",
        "entertainment industry",
        "education news",
        "crime and justice",
        "natural disasters",
        "climate change",
        "artificial intelligence",
        "space exploration",
        "renewable energy",
        "cybersecurity"
    ]
    
    queries = []
    for i in range(num_queries):
        if i < len(base_queries):
            queries.append(base_queries[i])
        else:
            base_query = random.choice(base_queries)
            variations = ["latest", "breaking", "important", "significant", "recent", "major"]
            variation = random.choice(variations)
            queries.append(f"{variation} {base_query}")
    
    return queries

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="mock_yes_or_no_requests.jsonl")
    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_article_chars", type=int, default=4000)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--rag_db_dir", type=str, default="./data")
    parser.add_argument("--rag_db_name", type=str, default="docs")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve per query")
    args = parser.parse_args()

    set_seeds(args.seed)

    print("Loading RAG system...")
    my_rag = rag.RAG(
        embedding_model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
        rag_searcher=rag.FAISSRAGSearcher(384),
        dimension=384,
        cache_size=1000,
        db_dir=args.rag_db_dir,
        db_name=args.rag_db_name,
        new_db=False  # Load existing database
    )

    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype="auto"
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    PROMPT_TEMPLATE = """You are generating **retrieval-focused yes/no questions** for a RAG benchmark.

    <context>
    {article}
    </context>

    Relevant supporting span:
    <span>
    {chunk}
    </span>

    Write ONE yes/no question that can be answered using ONLY the above span or immediate surrounding context.

    Strict rules (follow ALL):
    - The question must be fully answerable from the span. Do not ask about anything not explicitly stated.
    - The question must be a yes/no question. The answer must be exactly "yes" or "no" (lowercase).
    - Include at least two concrete anchors from the article (e.g., a PERSON/ORG and a DATE/NUMBER/LOCATION).
    - Avoid vague or abstract prompts (e.g., “main idea,” “impact,” “significance”).
    - The question must not be answerable from the title alone or general knowledge.
    - Use one sentence, 12–30 words.

    Output format:
    <question>your question here</question>
    <answer>yes or no</answer>
    Only output these tags and nothing else.
    """

    print("Generating retrieval queries...")
    retrieval_queries = generate_retrieval_queries(args.num_samples)
    
    print("Retrieving documents using RAG...")
    all_retrieved_docs = {}
    doc_count = 0
    
    for query in tqdm(retrieval_queries, desc="Retrieving documents"):
        if doc_count >= args.num_samples:
            break
        
        retrieved_docs = my_rag.retrieve([query], top_k=args.top_k)
        
        for doc_id, doc_info in retrieved_docs.items():
            if doc_id not in all_retrieved_docs and doc_count < args.num_samples:
                all_retrieved_docs[doc_id] = doc_info
                doc_count += 1
                if doc_count >= args.num_samples:
                    break

    print(f"Retrieved {len(all_retrieved_docs)} unique documents")

    out_path = args.output_file
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"Generating questions -> {out_path}")
    success_cnt = 0
    with open(out_path, "w", encoding="utf-8") as f_out:
        for idx, (doc_id, doc_info) in enumerate(tqdm(all_retrieved_docs.items(), total=len(all_retrieved_docs))):
            try:
                # article = doc_info["doc"]
                # article_trim = truncate_article(article, max_chars=args.max_article_chars)
                chunk = doc_info["doc"]
                article_id = doc_info["article_id"]
                article = my_rag.database.fetch_article(article_id)
                article_trim = truncate_article(article, max_chars=args.max_article_chars)

                prompt = PROMPT_TEMPLATE.format(article=article_trim, chunk=chunk)

                gen = generator(prompt)[0]["generated_text"]
                question = extract_question(gen)
                answer = extract_answer(gen)

                # make sure this is a question
                if not question.endswith("?"):
                    if re.search(r"^(who|what|when|where|why|how|which|did|does|do|is|are|was|were)\b", question, re.I):
                        question = question.rstrip(".") + "?"

                if not question:
                    pass # ignore this one

                record = {
                    "id": int(doc_id),
                    "article_id": article_id,
                    "question": question,
                    "answer": answer,
                    "content": article,
                    "chunk": chunk,
                }
                success_cnt += 1
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                # record = {
                #     "id": int(doc_id),
                #     "question": "What key event does this article describe?",
                #     "content": doc_info["doc"],
                #     "error": str(e),
                # }
                # f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(e)
                pass # skip this examples

    print(f"Successfully generated {success_cnt} questions out of {args.num_samples} requests")
    print("Done.")
