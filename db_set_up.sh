#!/bin/bash

python rag/database.py
python add_global_news_to_database.py
python add_multi_hop_to_database.py