import argparse
import os
import subprocess
import sys
import json
import pandas as pd
import tqdm
import rag.rag as rag
from rag.provider_factory import create_embedding_provider, create_vector_search_provider, create_llm_provider
from rag.aws_config import get_aws_region, get_bedrock_embedding_model, get_bedrock_llm_model
from pathlib import Path
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

def _initialize_rag_system(args):
    """Helper function to initialize the RAG system"""
    print(f"Loading RAG system from {args.db_dir}...")
    
    # Get AWS region
    aws_region = get_aws_region(args.aws_region)
    
    # Resolve model names for Bedrock
    if args.embedding_provider == "bedrock":
        embedding_model_name = get_bedrock_embedding_model(args.embedding_model)
    else:
        embedding_model_name = args.embedding_model
    
    # Create providers
    embedding_provider = create_embedding_provider(
        provider_type=args.embedding_provider,
        model_name=embedding_model_name,
        dimension=args.dimension,
        region_name=aws_region
    )
    
    vector_search_provider = create_vector_search_provider(
        provider_type=args.vector_search_provider,
        dimension=args.dimension,
        endpoint=args.opensearch_endpoint,
        region_name=aws_region
    )
    
    my_rag = rag.RAG(
        embedding_model=embedding_provider,
        rag_searcher=vector_search_provider,
        cache_size=args.cache_size,
        db_dir=args.db_dir,
        db_name=args.db_name,
    )
    
    return my_rag


def prepare_multihop_retrieval_data_from_dataset(args):
    print("Preparing retrieval data for MultiHop-RAG evaluation using official dataset")
    
    my_rag = _initialize_rag_system(args)
    
    # load MultiHop-RAG queries from the official dataset
    multihop_query_file = os.path.join(args.multihop_root, "dataset", "MultiHopRAG.json")
    if not os.path.exists(multihop_query_file):
        raise FileNotFoundError(
            f"MultiHop-RAG dataset not found at {multihop_query_file}. "
            f"Please ensure the MultiHop-RAG repository is properly set up."
        )
    
    print(f"Loading queries from {multihop_query_file}...")
    with open(multihop_query_file, 'r') as f:
        query_data = json.load(f)
    
    print(f"Loaded {len(query_data)} multi-hop queries from MultiHop-RAG dataset")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    pbar = tqdm.tqdm(query_data, desc="Retrieving documents")
    
    for data in pbar:
        query = data['query']
        
        retrieved = my_rag.retrieve([query], args.top_k)
        
        retrieval_list = []
        for chunk_id, chunk_info in retrieved.items():
            retrieval_list.append({
                "text": chunk_info['doc']
            })
        
        gold_list = []
        for evidence in data.get('evidence_list', []):
            gold_list.append({
                "fact": evidence['fact']
            })
        
        result = {
            "query": query,
            "question_type": data.get("question_type", "retrieval_query"),
            "retrieval_list": retrieval_list,
            "gold_list": gold_list,
        }
        results.append(result)
    
    json_path = output_dir / "multihop_retrieval_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved retrieval results to {json_path}")
    print(f"Total queries processed: {len(results)}")
    print(f"Average evidence facts per query: {sum(len(r['gold_list']) for r in results) / len(results):.2f}")
    
    return json_path


def prepare_multihop_retrieval_data(args):
    print("Preparing retrieval data for MultiHop-RAG evaluation")
    
    my_rag = _initialize_rag_system(args)
    
    print(f"Loading queries from {args.query_file}...")
    df = pd.read_json(args.query_file, lines=True)
    print(f"Loaded {len(df)} queries")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    pbar = tqdm.tqdm(df.iterrows(), total=len(df), desc="Retrieving documents")
    
    for idx, row in pbar:
        query = str(row["question"])
        try:
            row_id = row["id"]
            query_id = int(row_id)
            gt_chunk_id = int(row_id)  # ground truth chunk ID (1-indexed)
        except (KeyError, ValueError, TypeError):
            # resample
            query_id = int(idx) + 1 if isinstance(idx, (int, float)) else hash(idx) % 1000000
            gt_chunk_id = None
        try:
            gt_article_id = str(row["article_id"])
        except (KeyError, ValueError, TypeError):
            gt_article_id = None
        try:
            gt_chunk_text = str(row["chunk"])
        except (KeyError, ValueError, TypeError):
            gt_chunk_text = ""
        
        retrieved = my_rag.retrieve([query], args.top_k)
        
        # format for MultiHop-RAG evaluation
        # expected format: JSON file with list of dicts, each dict has:
        # - query: str
        # - question_type: str (default to "retrieval_query" if not specified)
        # - retrieval_list: list of dicts with 'text' field
        # - gold_list: list of dicts with 'fact' field
        
        # build retrieval_list (list of dicts with 'text' field)
        retrieval_list = []
        for chunk_id, chunk_info in retrieved.items():
            retrieval_list.append({
                "text": chunk_info['doc']
            })
        
        # build gold_list (list of dicts with 'fact' field)
        # the gold_list should contain the ground truth facts/chunks
        gold_list = []
        if gt_chunk_text:
            gold_list.append({
                "fact": gt_chunk_text
            })
        
        result = {
            "query": query,
            "question_type": row.get("question_type", "retrieval_query"),  # default type
            "retrieval_list": retrieval_list,
            "gold_list": gold_list,
        }
        results.append(result)
    
    json_path = output_dir / "multihop_retrieval_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved retrieval results to {json_path}")
    
    return json_path


def generate_qa_answers_from_dataset(args):
    print("Generating Q&A answers for MultiHop-RAG evaluation")
    
    my_rag = _initialize_rag_system(args)
    
    # Get AWS region
    aws_region = get_aws_region(args.aws_region)
    
    # Resolve model names for Bedrock
    if args.llm_provider == "bedrock":
        llm_model_name = get_bedrock_llm_model(args.llm_model)
    else:
        llm_model_name = args.llm_model
    
    print(f"Loading LLM model {llm_model_name}...")
    llm_provider = create_llm_provider(
        provider_type=args.llm_provider,
        model_name=llm_model_name,
        region_name=aws_region
    )
    
    multihop_query_file = os.path.join(args.multihop_root, "dataset", "MultiHopRAG.json")
    if not os.path.exists(multihop_query_file):
        raise FileNotFoundError(
            f"MultiHop-RAG dataset not found at {multihop_query_file}. "
            f"Please ensure the MultiHop-RAG repository is properly set up."
        )
    
    print(f"Loading queries from {multihop_query_file}...")
    with open(multihop_query_file, 'r') as f:
        query_data = json.load(f)
    
    print(f"Loaded {len(query_data)} multi-hop queries from MultiHop-RAG dataset")
    
    retrieval_results_path = None
    if args.retrieval_results and os.path.exists(args.retrieval_results):
        print(f"Loading retrieval results from {args.retrieval_results}...")
        with open(args.retrieval_results, 'r') as f:
            retrieval_data = {item['query']: item for item in json.load(f)}
        retrieval_results_path = Path(args.retrieval_results)
    else:
        print("No retrieval results provided. Will retrieve documents on the fly.")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    qa_prompt_template = """
You are a question-answering system. Your task is to answer the question strictly using the information provided in the context. 

Requirements:
- The answer must be a single word or entity.  
- If the context does not provide enough information, output exactly: Insufficient Information.  
- Do NOT provide explanations or reasoning.  
- Your entire answer must be wrapped in: <box> ... </box>  

Question:
{query}

Context:
{context}

Final Answer (in <box></box>):
"""

    
    results = []
    pbar = tqdm.tqdm(query_data, desc="Generating Q&A answers")
    
    for data in pbar:
        query = data['query']
        
        if retrieval_results_path and query in retrieval_data:
            retrieval_list = retrieval_data[query]['retrieval_list']
        else:
            retrieved = my_rag.retrieve([query], args.top_k)
            retrieval_list = []
            for chunk_id, chunk_info in retrieved.items():
                retrieval_list.append({
                    "text": chunk_info['doc']
                })
        
        context = '\n--------------\n'.join([item['text'] for item in retrieval_list])
        
        prompt = qa_prompt_template.format(query=query, context=context)
        
        try:
            model_answer = llm_provider.generate(
                prompt,
                max_new_tokens=args.qa_max_tokens,
                temperature=args.qa_temperature,
                top_p=args.qa_top_p
            )
            
            # clean up the answer (remove prompt if it was included)
            if prompt in model_answer:
                model_answer = model_answer.replace(prompt, "").strip()
            
            # import re
            # match = re.search(r'The answer to the question is "(.*?)"', model_answer)
            # if match:
            # model_answer = match.group(1)
            model_answer = extract_box(model_answer)
        except Exception as e:
            print(f"\nWarning: Error generating answer for query '{query[:50]}...': {e}")
            model_answer = "Error generating answer"
        
        result = {
            "query": query,
            "model_answer": model_answer,
            "question_type": data.get("question_type", "retrieval_query"),
            "gold_answer": data.get("answer", ""),
        }
        results.append(result)
    
    json_path = output_dir / "multihop_qa_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Q&A results to {json_path}")
    print(f"Total queries processed: {len(results)}")
    
    return json_path


def run_step(name: str, cmd: list, args: argparse.Namespace):
    """Run a MultiHop-RAG evaluation step"""
    print(f"MultiHop Running {name}")
    print(f"MultiHop Command: {' '.join(cmd)}")
    
    # run from project root, not from multihop_root to avoid path issues
    proc = subprocess.run(cmd, capture_output=True, text=True)

    print(f"\nMultiHop {name} OUT")
    print(proc.stdout.strip())

    if proc.stderr.strip():
        print(f"\nMultiHop ERR")
        print(proc.stderr.strip())

    if proc.returncode != 0:
        print(f"\nMultiHop {name} FAILED with exit code {proc.returncode}")
        return False
    else:
        print(f"\nMultiHop {name} COMPLETED successfully.")
        return True


def main(args):
    retrieval_script = os.path.join(args.multihop_root, "retrieval_evaluate.py")
    qa_script = os.path.join(args.multihop_root, "qa_evaluate.py")
    
    success = True

    if args.do_retrieval:
        # prepare retrieval data or use provided file
        if args.retrieval_results and os.path.exists(args.retrieval_results):
            retrieval_results_path = Path(args.retrieval_results)
            print(f"\nUsing existing retrieval results: {retrieval_results_path}")
        else:
            if args.use_multihop_dataset:
                retrieval_results_path = prepare_multihop_retrieval_data_from_dataset(args)
            else:
                retrieval_results_path = prepare_multihop_retrieval_data(args)
        
        # check if MultiHop-RAG repo exists
        if not os.path.exists(args.multihop_root):
            print(f"\nWARNING: MultiHop-RAG repository not found at {args.multihop_root}")
            print("Please update the submodule:")
            print(f"  cd {args.multihop_root}; git submodule update --init --recursive;")
            print("\n Skipping retrieval evaluation.")
            success = False
        elif not os.path.exists(retrieval_script):
            print(f"\nWARNING: Retrieval eval script not found: {retrieval_script}")
            print("Please check the MultiHop-RAG repository structure.")
            print("\nSkipping retrieval evaluation.")
            success = False
        else:
            # Use the correct command format based on the script
            # Use absolute path for the results file
            abs_results_path = os.path.abspath(str(retrieval_results_path))
            cmd = [args.python, retrieval_script, "--file", abs_results_path]
            cmd_worked = run_step("Retrieval Evaluation", cmd, args)
            if not cmd_worked:
                print(f"\n Evaluation failed. Please check the output above.")
                success = False

    # for Q&A questions
    if args.do_qa:
        if not os.path.exists(args.multihop_root):
            print(f"\nWARNING: MultiHop-RAG repository not found at {args.multihop_root}")
            print("Skipping QA evaluation.")
            success = False
        elif not os.path.exists(qa_script):
            print(f"\nWARNING: QA eval script not found: {qa_script}")
            print("Skipping QA evaluation.")
            success = False
        else:
            # Generate Q&A results if not provided
            if args.qa_results and os.path.exists(args.qa_results):
                qa_results_path = Path(args.qa_results)
                print(f"\nUsing existing QA results: {qa_results_path}")
            else:
                # Generate Q&A answers
                if args.use_multihop_dataset:
                    qa_results_path = generate_qa_answers_from_dataset(args)
                else:
                    print(f"\nWARNING: QA results file not found: {args.qa_results}")
                    print("Please generate QA results first or provide the correct path.")
                    print("Note: Q&A generation from custom query files is not yet implemented.")
                    print("Skipping QA evaluation.")
                    success = False
                    qa_results_path = None
            
            if qa_results_path:
                # The qa_evaluate.py script expects the file to be in qa_output/llama.json
                # and reads dataset/MultiHopRAG.json, so we need to run it from multihop_root
                qa_output_dir = os.path.join(args.multihop_root, "qa_output")
                os.makedirs(qa_output_dir, exist_ok=True)
                target_qa_file = os.path.join(qa_output_dir, "llama.json")
                
                # Copy our results to the expected location
                import shutil
                shutil.copy2(str(qa_results_path), target_qa_file)
                print(f"Copied QA results to {target_qa_file} for evaluation")
                
                # qa_evaluate.py doesn't accept command line arguments - it reads from hardcoded paths
                # So we need to run it from the multihop_root directory
                original_cwd = os.getcwd()
                try:
                    os.chdir(args.multihop_root)
                    # qa_evaluate.py reads from qa_output/llama.json and dataset/MultiHopRAG.json
                    cmd = [args.python, "qa_evaluate.py"]
                    cmd_worked = run_step("QA Evaluation", cmd, args)
                    if not cmd_worked:
                        print(f"\nQA evaluation failed. Please check the output above.")
                        success = False
                finally:
                    os.chdir(original_cwd)
    
    if success:
        print("All evaluations completed successfully!")
    else:
        print("Some evaluations had issues. Please check the output above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query_file",
        type=str,
        default="mock_requests.jsonl",
        help="Path to query file (JSONL format)",
    )
    parser.add_argument(
        "--multihop_root",
        type=str,
        default="MultiHop-RAG",
        help="Path to cloned yixuantt/MultiHop-RAG repo",
    )
    parser.add_argument(
        "--retrieval_results",
        type=str,
        default=None,
        help="Path to existing retrieval results JSON file (if not provided, will generate from query_file)",
    )
    parser.add_argument(
        "--qa_results",
        type=str,
        default="multihop_qa_results.jsonl",
        help="Path to QA results file compatible with MultiHop-RAG qa_evaluate.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="multihop_output",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        default="./data",
        help="RAG database directory",
    )
    parser.add_argument(
        "--db_name",
        type=str,
        default="docs",
        help="RAG database name",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of documents to retrieve per query",
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=1000,
        help="LRU cache size for RAG system",
    )
    parser.add_argument(
        "--do_retrieval",
        action="store_true",
        help="Run retrieval evaluation",
    )
    parser.add_argument(
        "--do_qa",
        action="store_true",
        help="Run QA evaluation",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use when invoking MultiHop-RAG scripts",
    )
    parser.add_argument(
        "--embedding_provider",
        type=str,
        default="sentence_transformer",
        choices=["sentence_transformer", "bedrock"],
        help="Embedding provider type",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name",
    )
    parser.add_argument(
        "--vector_search_provider",
        type=str,
        default="faiss",
        choices=["faiss", "opensearch"],
        help="Vector search provider type",
    )
    parser.add_argument(
        "--aws_region",
        type=str,
        default=None,
        help="AWS region (defaults to AWS_REGION env var or us-east-1)",
    )
    parser.add_argument(
        "--opensearch_endpoint",
        type=str,
        default=None,
        help="OpenSearch endpoint URL (required for opensearch provider)",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=384,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--use_multihop_dataset",
        action="store_true",
        help="Use official MultiHop-RAG dataset queries instead of custom query_file. "
             "This enables proper multi-hop evaluation with 2-4 evidence facts per query.",
    )
    parser.add_argument(
        "--llm_provider",
        type=str,
        default="huggingface",
        choices=["huggingface", "bedrock"],
        help="LLM provider type for Q&A generation",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="LLM model name for Q&A generation",
    )
    parser.add_argument(
        "--qa_max_tokens",
        type=int,
        default=512,
        help="Maximum tokens for Q&A answer generation",
    )
    parser.add_argument(
        "--qa_temperature",
        type=float,
        default=0.1,
        help="Temperature for Q&A answer generation",
    )
    parser.add_argument(
        "--qa_top_p",
        type=float,
        default=0.95,
        help="Top-p for Q&A answer generation",
    )
    args = parser.parse_args()

    # If neither flag is set, run retrieval evaluation (QA requires separate results)
    if not args.do_retrieval and not args.do_qa:
        args.do_retrieval = True

    main(args)
    