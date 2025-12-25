#!/bin/bash
# Install missing dependencies in the background if needed
pip install streamlit pandas psycopg2-binary pgvector sentence-transformers python-frontmatter uuid-utils requests torch --index-url https://download.pytorch.org/whl/cpu --user

# Run Streamlit and worker
python3 src/worker.py &
python3 -m streamlit run src/app.py --server.port=8505 --server.address=0.0.0.0
