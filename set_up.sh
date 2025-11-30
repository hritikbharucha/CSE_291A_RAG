#!/bin/bash
export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_SESSION_TOKEN=

export AWS_REGION="us-west-2"

echo "AWS Bedrock credentials configured."

echo "Adding MultiHop-RAG submodule..."
cd MultiHop-RAG; git submodule update --init --recursive; cd ..;
python rag/database.py
echo "MultiHop-RAG submodule added."

echo "Setting up database..."
python add_global_news_to_database.py \
    --embedding_provider sentence_transformer \

python add_multi_hop_to_database.py \
    --embedding_provider sentence_transformer \

echo "Database setup complete!"