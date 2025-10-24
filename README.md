# CSE 291A RAG Project

This project implements a Retrieval-Augmented Generation (RAG) system for question generation and testing using news articles.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up the database:
```bash
bash db_set_up.sh
```

## Usage

### Step 1: Generate Questions
Generate retrieval-targeted questions from news articles:
```bash
python generate_questions.py
```

Optional parameters:
- `--output_file`: Output file path (default: mock_requests.jsonl)
- `--num_samples`: Number of samples to generate (default: 100)
- `--model_name`: LLM model to use (default: mistralai/Mistral-7B-Instruct-v0.2)
- `--rag_db_dir`: RAG database directory (default: ./data)
- `--rag_db_name`: RAG database name (default: docs)

### Step 2: Generate Yes/No Questions
Generate yes/no questions with answers:
```bash
python generate_yes_or_no_questions.py
```

Optional parameters:
- `--output_file`: Output file path (default: mock_yes_or_no_requests.jsonl)
- `--num_samples`: Number of samples to generate (default: 100)
- `--model_name`: LLM model to use (default: mistralai/Mistral-7B-Instruct-v0.2)
- `--rag_db_dir`: RAG database directory (default: ./data)
- `--rag_db_name`: RAG database name (default: docs)

### Step 3: Test the System
Test retrieval accuracy and quality:
```bash
python test.py
```

Test modes:
- `--mode retrieval_accuracy`: Test retrieval accuracy (default)
- `--mode retrieval_quality`: Test retrieval quality with LLM

Optional parameters:
- `--query_file`: Input query file (default: mock_requests.jsonl)
- `--llm`: LLM model to use (default: mistralai/Mistral-7B-Instruct-v0.2)

## Project Structure

- `rag/`: RAG system implementation
  - `rag.py`: Main RAG class
  - `database.py`: SQLite database operations
  - `searcher.py`: FAISS-based document search
  - `lru_cache.py`: LRU cache implementation
- `generate_questions.py`: Generate open-ended questions
- `generate_yes_or_no_questions.py`: Generate yes/no questions
- `test.py`: Test retrieval accuracy and quality
- `add_global_news_to_database.py`: Add news articles to database
- `db_set_up.sh`: Database setup script

## Requirements

- Python 3.11+
- PyTorch
- Transformers
- Sentence Transformers
- FAISS
- SQLite3
- Pandas
- NumPy