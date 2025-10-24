import argparse
import os
import rag.rag as rag
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import tqdm
import re

def extract_box(text: str) -> str:
    matches = re.findall(r"<box>\s*(.*?)\s*</box>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        # return the first match
        return matches[-1].strip()

    cleaned = re.sub(r"<[^>]+>", "", text).strip()
    for line in cleaned.splitlines():
        s = line.strip()
        if s:
            return s
    return cleaned

def clean_letters_only(text: str) -> str:
    # Keep only letters and spaces
    letters_only = re.sub(r'[^a-zA-Z\s]', '', text)
    # Replace multiple spaces with a single space and strip leading/trailing spaces
    cleaned = re.sub(r'\s+', ' ', letters_only).strip()
    return cleaned

def test_retrieval_accuracy():
    df = pd.read_json(args.query_file, lines=True)
    top1_acc = 0
    top5_acc = 0
    for index, row in tqdm.tqdm(df.iterrows()):
        query = row["question"]
        gt_idx = row["id"] - 1

        top1_rslts = my_rag.retrieve([query], 1)
        top5_rslts = my_rag.retrieve([query], 5)
        top1_acc += 1 if gt_idx in top1_rslts.keys() else 0
        top5_acc += 1 if gt_idx in top5_rslts.keys() else 0

        # print(f"gt_id: {gt_idx}, query: {query}, top1_rslt: {top1_rslts.keys()}, top5_rslts: {top5_rslts.keys()}")

    print(f"Top-1 acc: {top1_acc / len(df)}, Top-5 acc: {top5_acc / len(df)}")

def test_retrieval_quality(topk=1):
    print(f"Loading model {args.llm}...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.llm,
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

    PROMPT_TEMPLATE = """
    User: You are given the following news article. Please answer the question based only on the information provided in the article. 
    And provide your final answer inside <box> ... </box> where it can only be "yes" or "no" in lower letters.

    <context>
    {article}
    </context>
    
    Question: {question}
    
    Answer:
    """
    df = pd.read_json(args.query_file, lines=True)
    accuracy = 0
    for index, row in tqdm.tqdm(df.iterrows()):
        query = row["question"]
        answer = row["answer"]

        article = my_rag.retrieve([query], topk)
        article = "/n".join([article[key]['doc'] for key in article.keys()]) # in case we have topk > 1 article
        prompt = PROMPT_TEMPLATE.format(article=article, question=query)
        gen = generator(prompt)[0]["generated_text"]
        llm_answer = extract_box(gen)

        answer = clean_letters_only(answer).lower()
        llm_answer = clean_letters_only(llm_answer).lower()
        accuracy += 1 if answer == llm_answer else 0
    print(f"Accuracy: {accuracy / len(df)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['retrieval_accuracy', 'retrieval_quality'], default='retrieval_accuracy')
    parser.add_argument('--llm', type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument('--query_file', type=str, default="mock_requests.jsonl")
    parser.add_argument("--max_article_chars", type=int, default=4000)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    my_rag = rag.RAG(
        embedding_model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
        rag_searcher=rag.FAISSRAGSearcher(384),
        dimension=384,
        cache_size=1000,
        db_dir="./data",
        db_name="docs",
    )

    if args.mode == "retrieval_accuracy":
        test_retrieval_accuracy()
    elif args.mode == "retrieval_quality":
        test_retrieval_quality()
    else:
        raise Exception(f"Unknown mode {args.mode}")


