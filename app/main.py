import sys
import os

# Ensure project root is in path so 'pipelines' and 'app' are importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.grader:app", host="0.0.0.0", port=8000, reload=False)
