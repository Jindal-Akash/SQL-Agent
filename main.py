from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from agent import ask

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.get("/")
def home():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        result = ask(req.message, req.session_id)
    except Exception as e:
        print("chat error:", e)  # full error stays in the server log
        return JSONResponse({"error": "The AI service had a problem. Please try again."},
                            status_code=500)
    return {"sql": result["sql"], "reply": result["reply"],
            "columns": result["columns"], "rows": result["rows"]}