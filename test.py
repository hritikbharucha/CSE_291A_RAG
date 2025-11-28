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
    chunk_top1_acc = 0
    chunk_top5_acc = 0
    article_top1_acc = 0
    article_top5_acc = 0
    total = 0
    pbar = tqdm.tqdm(df.iterrows(), total=len(df))
    for index, row in pbar:
        query = row["question"]
        gt_idx = row["id"]  # Database IDs are 1-indexed, no need to subtract 1
        gt_article_id = row["article_id"]

        top1_rslts = my_rag.retrieve([query], 1)
        top5_rslts = my_rag.retrieve([query], 5)

        chunk_top1_acc += 1 if gt_idx in top1_rslts.keys() else 0
        chunk_top5_acc += 1 if gt_idx in top5_rslts.keys() else 0

        top1_article_id = [top1_rslts[key]["article_id"] for key in top1_rslts]
        top5_article_id = [top5_rslts[key]["article_id"] for key in top5_rslts]
        article_top1_acc += 1 if gt_article_id in top1_article_id else 0
        article_top5_acc += 1 if gt_article_id in top5_article_id else 0

        total += 1

        # print(f"gt_id: {gt_idx}, query: {query}, top1_rslt: {top1_rslts.keys()}, top5_rslts: {top5_rslts.keys()}")
        # print(f"gt_article_id: {gt_article_id}, query: {query}, "
        #       f"article_top1_rslt: {top1_article_id}, article_top5_rslts: {top5_article_id}")
        pbar.set_description(
            desc=f"top1_acc: {chunk_top1_acc/total:.4f}, top5_acc: {chunk_top5_acc/total:.4f} "
                 f"top1_acc_article: {article_top1_acc/total:.4f}, top5_acc_article: {article_top5_acc/total:.4f}")

    print(f"Top-1 acc: {chunk_top1_acc / len(df)}, Top-5 acc: {chunk_top5_acc / len(df)}, Article Top-1 acc: {article_top1_acc/total}, Top-5 acc: {article_top5_acc/total}")

def test_retrieval_quality_baseline():
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
    User: You are given the question, and provide your final answer inside <box> ... </box> where it can only be "yes" or "no" in lower letters.

    Question: {question}

    Answer:
    """
    df = pd.read_json(args.query_file, lines=True)

    accuracy = 0
    total = 0

    tp = fp = fn = tn = 0
    invalid_preds = 0

    pbar = tqdm.tqdm(df.iterrows(), total=len(df))
    for index, row in pbar:
        query = row["question"]
        answer = row["answer"]

        prompt = PROMPT_TEMPLATE.format(question=query)
        gen = generator(prompt)[0]["generated_text"]
        llm_answer = extract_box(gen)

        answer = clean_letters_only(answer).lower()
        llm_answer = clean_letters_only(llm_answer).lower()

        accuracy += 1 if answer == llm_answer else 0
        total += 1

        if llm_answer in {"yes", "no"}:
            if answer == "yes" and llm_answer == "yes":
                tp += 1
            elif answer == "no" and llm_answer == "yes":
                fp += 1
            elif answer == "yes" and llm_answer == "no":
                fn += 1
            elif answer == "no" and llm_answer == "no":
                tn += 1
        else:
            invalid_preds += 1

        pbar.set_description(desc=f"accuracy: {accuracy / total:.4f}")

    acc = accuracy / len(df)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"Accuracy:  {acc:.6f}  (over all {len(df)} examples)")
    print(f"Precision: {precision:.6f}  (pos='yes')")
    print(f"Recall:    {recall:.6f}  (pos='yes')")
    print(f"F1:        {f1:.6f}      (pos='yes')")
    print(f"Confusion Matrix (valid preds only): TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    if invalid_preds:
        print(f"Note: {invalid_preds} predictions were invalid (not 'yes'/'no') and excluded from PR/F1.")

def test_retrieval_quality(topk=5):
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
    total = 0

    tp = fp = fn = tn = 0
    invalid_preds = 0

    pbar = tqdm.tqdm(df.iterrows(), total=len(df))
    for index, row in pbar:
        query = row["question"]
        answer = row["answer"]

        article = my_rag.retrieve([query], topk)
        article = "\n".join([article[key]['doc'] for key in article.keys()])  # keep as-is
        prompt = PROMPT_TEMPLATE.format(article=article, question=query)
        gen = generator(prompt)[0]["generated_text"]
        llm_answer = extract_box(gen)

        answer = clean_letters_only(answer).lower()
        llm_answer = clean_letters_only(llm_answer).lower()

        accuracy += 1 if answer == llm_answer else 0
        total += 1

        if llm_answer in {"yes", "no"}:
            if answer == "yes" and llm_answer == "yes":
                tp += 1
            elif answer == "no" and llm_answer == "yes":
                fp += 1
            elif answer == "yes" and llm_answer == "no":
                fn += 1
            elif answer == "no" and llm_answer == "no":
                tn += 1
        else:
            invalid_preds += 1

        pbar.set_description(desc=f"accuracy: {accuracy / total:.4f}")

    acc = accuracy / len(df)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"Accuracy:  {acc:.6f}  (over all {len(df)} examples)")
    print(f"Precision: {precision:.6f}  (pos='yes')")
    print(f"Recall:    {recall:.6f}  (pos='yes')")
    print(f"F1:        {f1:.6f}      (pos='yes')")
    print(f"Confusion Matrix (valid preds only): TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    if invalid_preds:
        print(f"Note: {invalid_preds} predictions were invalid (not 'yes'/'no') and excluded from PR/F1.")


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
        cache_size=0,
        db_dir="./data",
        db_name="docs",
    )

    if args.mode == "retrieval_accuracy":
        test_retrieval_accuracy()
    elif args.mode == "retrieval_quality":
        test_retrieval_quality_baseline()
        test_retrieval_quality()
    else:
        raise Exception(f"Unknown mode {args.mode}")


