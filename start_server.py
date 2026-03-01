"""
Run this from the project root: python start_server.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import uvicorn

if __name__ == "__main__":
    print("Starting Ankya AI backend on http://localhost:8000 ...")
    uvicorn.run("app.grader:app", host="0.0.0.0", port=8000, reload=False)
