"""
FraudShield API — FastAPI backend for CSV fraud detection pipeline.
Deployed on Railway. No LLM API keys — uses scikit-learn + XGBoost only.
"""

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from pipeline import process_csv

app = FastAPI(
    title="FraudShield API",
    description="End-to-end fraud detection pipeline. Upload CSV, get analysis.",
    version="1.0.0",
)

# Allow all origins (frontend on Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok", "service": "FraudShield API", "engine": "XGBoost + RandomForest + GradientBoosting"}


@app.post("/api/analyze")
async def analyze_csv(request: Request):
    """Accept raw CSV text in POST body and run the fraud detection pipeline."""
    body = await request.body()
    csv_text = body.decode("utf-8")

    if len(csv_text) < 50:
        return JSONResponse(status_code=400, content={"error": "CSV too small or empty"})

    if len(csv_text) > 50 * 1024 * 1024:  # 50MB limit
        return JSONResponse(status_code=413, content={"error": "CSV exceeds 50MB limit"})

    result = process_csv(csv_text)

    if "error" in result:
        return JSONResponse(status_code=500, content=result)

    return JSONResponse(content=result)


@app.post("/api/analyze-file")
async def analyze_csv_file(file: UploadFile = File(...)):
    """Accept CSV file upload and run the fraud detection pipeline."""
    contents = await file.read()
    csv_text = contents.decode("utf-8")

    if len(csv_text) < 50:
        return JSONResponse(status_code=400, content={"error": "CSV too small or empty"})

    result = process_csv(csv_text)

    if "error" in result:
        return JSONResponse(status_code=500, content=result)

    return JSONResponse(content=result)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
