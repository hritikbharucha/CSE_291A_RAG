import argparse
import os
import rag.rag as rag
from rag.provider_factory import create_embedding_provider, create_vector_search_provider, create_llm_provider
from rag.aws_config import get_aws_region, get_bedrock_embedding_model, get_bedrock_llm_model
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

def normalize_for_containment(text: str) -> str:
    """Normalize text for containment checks (lowercase, no whitespace)."""
    if not text:
        return ""
    return re.sub(r'\s+', '', str(text)).lower()

def tokenize_to_set(text: str) -> set:
    """Tokenize text into a set of alphanumeric lowercase tokens."""
    if not text:
        return set()
    return set(re.findall(r'\w+', str(text).lower()))

def token_iou(tokens_a: set, tokens_b: set) -> float:
    """Compute IoU between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    inter = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(inter) / len(union) if union else 0.0

def chunk_match_score(target_tokens: set, target_norm: str, text: str) -> float:
    """Return max of token IoU and full-containment indicator."""
    tokens = tokenize_to_set(text)
    norm = normalize_for_containment(text)
    iou = token_iou(target_tokens, tokens)
    contains = 1.0 if norm and target_norm and norm in target_norm else 0.0
    return max(iou, contains)

def test_retrieval_accuracy(topk: int = 5):
    df = pd.read_json(args.query_file, lines=True)
    chunk_top1_acc = 0
    chunk_topk_acc = 0
    article_top1_acc = 0
    article_topk_acc = 0
    total = 0
    pbar = tqdm.tqdm(df.iterrows(), total=len(df))
    for index, row in pbar:
        query = row["question"]
        gt_idx = row["id"]  # Database IDs are 1-indexed, no need to subtract 1
        gt_article_id = row["article_id"]
        gt_chunk_text = row.get("chunk", "")
        target_tokens = tokenize_to_set(gt_chunk_text) if args.chunk_text_match else set()
        target_norm = normalize_for_containment(gt_chunk_text) if args.chunk_text_match else ""

        top1_rslts = my_rag.retrieve([query], 1)
        topk_rslts = my_rag.retrieve([query], topk)

        if args.chunk_text_match:
            max_score_top1 = max(
                (chunk_match_score(target_tokens, target_norm, top1_rslts[key]["doc"]) for key in top1_rslts),
                default=0.0,
            )
            max_score_topk = max(
                (chunk_match_score(target_tokens, target_norm, topk_rslts[key]["doc"]) for key in topk_rslts),
                default=0.0,
            )
            chunk_top1_acc += 1 if max_score_top1 >= args.chunk_iou_threshold else 0
            chunk_topk_acc += 1 if max_score_topk >= args.chunk_iou_threshold else 0
        else:
            chunk_top1_acc += 1 if gt_idx in top1_rslts.keys() else 0
            chunk_topk_acc += 1 if gt_idx in topk_rslts.keys() else 0

        top1_article_id = [top1_rslts[key]["article_id"] for key in top1_rslts]
        topk_article_id = [topk_rslts[key]["article_id"] for key in topk_rslts]
        article_top1_acc += 1 if gt_article_id in top1_article_id else 0
        article_topk_acc += 1 if gt_article_id in topk_article_id else 0

        total += 1

        # print(f"gt_id: {gt_idx}, query: {query}, top1_rslt: {top1_rslts.keys()}, top5_rslts: {top5_rslts.keys()}")
        # print(f"gt_article_id: {gt_article_id}, query: {query}, "
        #       f"article_top1_rslt: {top1_article_id}, article_top5_rslts: {top5_article_id}")
        desc = (
            f"top1_acc: {chunk_top1_acc/total:.4f}, top{topk}_acc: {chunk_topk_acc/total:.4f} "
            f"top1_acc_article: {article_top1_acc/total:.4f}, top{topk}_acc_article: {article_topk_acc/total:.4f}"
        )
        pbar.set_description(desc=desc)

    print(
        f"Top-1 acc: {chunk_top1_acc / len(df)}, "
        f"Top-{topk} acc: {chunk_topk_acc / len(df)}, "
        f"Article Top-1 acc: {article_top1_acc/total}, "
        f"Article Top-{topk} acc: {article_topk_acc/total}"
    )

def test_retrieval_quality_baseline():
    print(f"Loading LLM model {args.llm}...")
    llm_provider = create_llm_provider(
        provider_type=args.llm_provider,
        model_name=args.llm_model_name,
        region_name=args.aws_region
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
        gen = llm_provider.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
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
    print(f"Loading LLM model {args.llm}...")
    llm_provider = create_llm_provider(
        provider_type=args.llm_provider,
        model_name=args.llm_model_name,
        region_name=args.aws_region
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
        gen = llm_provider.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
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
    parser.add_argument(
        "--chunk_text_match",
        action="store_true",
        help="When set, treat a retrieval as correct if its text overlaps the ground truth 'chunk' above the IoU/containment threshold.",
    )
    parser.add_argument(
        "--chunk_iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold (0-1) for considering retrieved chunk text a correct match when --chunk_text_match is set.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Top-k to use when evaluating retrieval accuracy.",
    )
    
    # Provider arguments
    parser.add_argument("--embedding_provider", type=str, default="sentence_transformer",
                       choices=["sentence_transformer", "bedrock"],
                       help="Embedding provider type")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Embedding model name")
    parser.add_argument("--vector_search_provider", type=str, default="faiss",
                       choices=["faiss", "opensearch", "hybrid"],
                       help="Vector search provider type")
    parser.add_argument("--llm_provider", type=str, default="huggingface",
                       choices=["huggingface", "bedrock"],
                       help="LLM provider type")
    parser.add_argument("--aws_region", type=str, default=None,
                       help="AWS region (defaults to AWS_REGION env var or us-east-1)")
    parser.add_argument("--opensearch_endpoint", type=str, default=None,
                       help="OpenSearch endpoint URL (required for opensearch provider)")
    parser.add_argument("--dimension", type=int, default=384,
                       help="Embedding dimension")
    parser.add_argument("--sparse_type", type=str, default="bm25",
                       choices=["tfidf", "bm25"],
                       help="Sparse backend for hybrid search")
    parser.add_argument("--hybrid_alpha", type=float, default=0.5,
                       help="Weight for sparse scores when using hybrid search")
    parser.add_argument("--db_dir", type=str, default="./data",
                       help="RAG database directory")
    parser.add_argument("--db_name", type=str, default="docs",
                       help="RAG database name")
    
    args = parser.parse_args()
    
    # Get AWS region
    aws_region = get_aws_region(args.aws_region)
    
    # Resolve model names for Bedrock
    if args.embedding_provider == "bedrock":
        embedding_model_name = get_bedrock_embedding_model(args.embedding_model)
    else:
        embedding_model_name = args.embedding_model
    
    if args.llm_provider == "bedrock":
        llm_model_name = get_bedrock_llm_model(args.llm)
    else:
        llm_model_name = args.llm
    
    args.llm_model_name = llm_model_name
    args.aws_region = aws_region

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
        cache_size=0,
        db_dir=args.db_dir,
        db_name=args.db_name,
    )

    if args.mode == "retrieval_accuracy":
        topk = max(1, args.topk)
        test_retrieval_accuracy(topk=topk)
    elif args.mode == "retrieval_quality":
        # test_retrieval_quality_baseline()
        test_retrieval_quality()
    else:
        raise Exception(f"Unknown mode {args.mode}")


