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

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_box(text: str) -> str:
    m = re.search(r"<box>\s*(.*?)\s*</box>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default="./data/GlobalNewsDataset/data.csv")
    parser.add_argument('--output_file', type=str, default="global_news_requests.jsonl")
    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_article_chars", type=int, default=4000)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    set_seeds(args.seed)

    print("Loading dataset...")
    df = pd.read_csv(args.csv_path)
    df = df.dropna(subset=["full_content"])
    # Ensure reproducible sampling order without altering your signature/vars
    articles_all = df["full_content"].tolist()
    indices = list(range(len(articles_all)))
    random.shuffle(indices)
    indices = indices[: args.num_samples]
    articles = [articles_all[i] for i in indices]

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

    # Prompt Template
    PROMPT_TEMPLATE = """You are creating **retrieval-targeted** questions for a RAG benchmark.

    Given this news article:
    <context>
    {article}
    </context>

    Write ONE realistic question that **requires retrieving a specific span** from the article.
    Strict rules (must follow all):
    - Include **at least two concrete anchors** from the article (e.g., a PERSON/ORG and a DATE/NUMBER/LOCATION).
    - The answer should be a **short text span** (a name, date, figure, place, or short clause), not an opinion or summary.
    - **No vague prompts** like "main idea", "impact", "implications", "significance", or "why is this important".
    - **Not answerable from the title alone** or from general knowledge; it must depend on details in the body text.
    - Use **one sentence**, 12–30 words.

    Output ONLY the question inside <box>...</box>, nothing else.
    """

    out_path = args.output_file
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print(f"Generating questions -> {out_path}")
    with open(out_path, "w", encoding="utf-8") as f_out:
        for idx, (global_idx, article) in enumerate(tqdm(zip(indices, articles), total=len(articles))):
            try:
                article_trim = truncate_article(article, max_chars=args.max_article_chars)
                prompt = PROMPT_TEMPLATE.format(article=article_trim)

                gen = generator(prompt)[0]["generated_text"]
                question = extract_box(gen)

                if not question.endswith("?"):
                    if re.search(r"^(who|what|when|where|why|how|which|did|does|do|is|are|was|were)\b", question, re.I):
                        question = question.rstrip(".") + "?"
                title = str(df.iloc[global_idx]["title"]) if "title" in df.columns else "the article"
                if not question:
                    question = f'In the article titling {title}, what key event does the author describe?'

                record = {
                    "id": idx,
                    "source_row": int(global_idx),
                    "question": question,
                    "context_excerpt": truncate_article(article, max_chars=400),
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                record = {
                    "id": idx,
                    "source_row": int(global_idx),
                    "question": "What key event does this article describe?",
                    "context_excerpt": truncate_article(article, max_chars=400),
                    "error": str(e),
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Done.")
