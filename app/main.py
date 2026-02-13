# app/main.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.grader:app", host="0.0.0.0", port=8000, reload=False)
