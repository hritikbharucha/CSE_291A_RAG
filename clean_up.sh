#!/bin/bash

# Clean up script for CSE 291A RAG Project
# This script removes all generated files and directories to start fresh

echo "Cleaning up generated files and directories..."

# Remove database files
echo "Removing database files..."
rm -f data/docs.sqlite
rm -f data/database.sql

# Remove index files
echo "Removing index files..."
rm -rf data/index/

# Remove generated question files
echo "Removing generated question files..."
rm -f mock_requests.jsonl
rm -f mock_yes_or_no_requests.jsonl
rm -f question_generation/global_news_requests.jsonl

# Remove output directories
echo "Removing output directories..."
rm -rf openrag_output/
rm -rf multihop_output/

# Remove other generated files
echo "Removing other generated files..."
rm -f retrieval_results.txt
rm -f openrag_eval_config.yaml

# Remove Python cache
echo "Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

echo "Cleanup complete!"
echo ""
echo "Note: Source datasets in data/GlobalNewsDataset/, data/MultiHopDataset/, and data/WikipediaSTEM270K/ are preserved."
echo "To rebuild the database, run: bash set_up.sh"

