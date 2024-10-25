#!/bin/bash
# start.sh

# Start uvicorn in the background
poetry run uvicorn app:app --host 0.0.0.0 --port 8000 &

# Start streamlit
poetry run streamlit run streaming.py --server.port 8501 --server.address 0.0.0.0