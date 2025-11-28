import argparse
import os
import subprocess
import sys
import json
import pandas as pd
import tqdm
import rag.rag as rag
from sentence_transformers import SentenceTransformer
from pathlib import Path


def prepare_multihop_retrieval_data(args):
    print("Preparing retrieval data for MultiHop-RAG evaluation")
    
    print(f"Loading RAG system from {args.db_dir}...")
    my_rag = rag.RAG(
        embedding_model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
        rag_searcher=rag.FAISSRAGSearcher(384),
        dimension=384,
        cache_size=args.cache_size,
        db_dir=args.db_dir,
        db_name=args.db_name,
    )
    
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
            print(f"\n[MultiHop] Using existing retrieval results: {retrieval_results_path}")
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
        elif not os.path.exists(args.qa_results):
            print(f"\nWARNING: QA results file not found: {args.qa_results}")
            print("Please generate QA results first or provide the correct path.")
            print("Skipping QA evaluation.")
            success = False
        else:
            # Try different possible command formats
            possible_cmds = [
                [args.python, qa_script, "--file", args.qa_results],
                [args.python, qa_script, "--input", args.qa_results],
                [args.python, qa_script, args.qa_results],
            ]
            
            cmd_worked = False
            for cmd in possible_cmds:
                cmd_worked = run_step("QA Evaluation", cmd, args)
                if cmd_worked:
                    break
            
            if not cmd_worked:
                print(f"\nCould not determine correct command format for {qa_script}")
                success = False
    
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
    args = parser.parse_args()

    # If neither flag is set, run retrieval evaluation (QA requires separate results)
    if not args.do_retrieval and not args.do_qa:
        args.do_retrieval = True

    main(args)
    