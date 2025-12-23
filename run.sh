#!/bin/bash
# Documentation App Startup Script

# Navigate to project directory
cd /home/ross/pythonproject/doc-manager/src

# Start background worker
python3 worker.py &
WORKER_PID=$!

# Start Streamlit
streamlit run app.py --server.port 8505 --server.address 0.0.0.0

# Cleanup on exit
kill $WORKER_PID
