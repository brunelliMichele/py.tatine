# server api di test (non serve all'esame)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Dict
import os

app = FastAPI()

@app.post("/submit")
async def receive_submission(submission: List[Dict]):
    # Validate structure
    for item in submission:
        if "filename" not in item or "samples" not in item:
            return JSONResponse(status_code=400, content={"error": "Invalid format"})
        if not isinstance(item["samples"], list) or len(item["samples"]) != 3:
            return JSONResponse(status_code=400, content={"error": "Each 'samples' must be a list of 3 items."})

    return {"message": "Submission received successfully", "num_entries": len(submission)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("mock_api_server:app", host="0.0.0.0", port=port, reload=True)