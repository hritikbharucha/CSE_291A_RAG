#!/bin/bash

cd MultiHop-RAG; git submodule update --init --recursive; cd ..;
python rag/database.py
python add_global_news_to_database.py
python add_multi_hop_to_database.py