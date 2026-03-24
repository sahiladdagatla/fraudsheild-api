"""
FraudShield API — FastAPI backend for CSV fraud detection pipeline.
Optimized for performance: GZip, multi-worker, thread pool, memory management.
"""

import asyncio
import gc
import os
import sys
import traceback

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, ORJSONResponse
import uvicorn

from pipeline import run_pipeline as _run_pipeline

# Pre-import heavy libraries at module load (not per-request)
import numpy as np
import pandas as pd
import sklearn
import xgboost

def process_csv(csv_text):
    """Wrapper with error handling and memory cleanup."""
    try:
        result = _run_pipeline(csv_text)
        gc.collect()  # Force garbage collection after heavy pipeline
        return result
    except Exception as e:
        gc.collect()
        return {"error": str(e), "traceback": traceback.format_exc()}

app = FastAPI(
    title="FraudShield API",
    description="End-to-end fraud detection pipeline. Upload CSV, get analysis.",
    version="2.0.0",
)

# GZip compression — reduces response size by ~70% (100KB → 30KB)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "FraudShield API",
        "version": "v6-optimized",
        "python": sys.version,
        "engine": "XGBoost + RandomForest + GradientBoosting",
    }


@app.post("/api/analyze")
async def analyze_csv(request: Request):
    """Accept raw CSV text in POST body and run the fraud detection pipeline.
    Runs in thread pool to avoid blocking the event loop."""
    body = await request.body()
    csv_text = body.decode("utf-8")

    if len(csv_text) < 50:
        return JSONResponse(status_code=400, content={"error": "CSV too small or empty"})

    if len(csv_text) > 50 * 1024 * 1024:
        return JSONResponse(status_code=413, content={"error": "CSV exceeds 50MB limit"})

    # Run in thread pool — prevents blocking the event loop
    # This is critical: sklearn/xgboost are CPU-bound and would freeze the server
    result = await asyncio.to_thread(process_csv, csv_text)

    if "error" in result:
        return JSONResponse(status_code=500, content=result)

    return JSONResponse(content=result)


@app.post("/api/analyze-file")
async def analyze_csv_file(file: UploadFile = File(...)):
    """Accept CSV file upload."""
    contents = await file.read()
    csv_text = contents.decode("utf-8")

    if len(csv_text) < 50:
        return JSONResponse(status_code=400, content={"error": "CSV too small or empty"})

    result = await asyncio.to_thread(process_csv, csv_text)

    if "error" in result:
        return JSONResponse(status_code=500, content=result)

    return JSONResponse(content=result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Use 2 workers on Railway/Render (512MB RAM allows it)
    # Each worker handles requests independently
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=2,
        timeout_keep_alive=300,  # 5 min keep-alive for long requests
    )
