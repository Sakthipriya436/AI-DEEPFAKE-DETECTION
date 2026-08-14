"""
Deepfake Detector Backend (FastAPI)
=====================================
Acts as the API gateway between the React frontend and the AI service.
Supports large file uploads (100MB+) via streaming.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import tempfile
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Deepfake Detector Backend",
    description="API gateway for the deepfake detection AI service",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:5000")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))


@app.get("/")
def read_root():
    return {"message": "Deepfake Detector Backend v2.0 running", "ai_service": AI_SERVICE_URL}


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{AI_SERVICE_URL}/health")
            ai_status = r.json()
    except Exception as e:
        ai_status = {"status": "unreachable", "error": str(e)}
    return {"status": "healthy", "ai_service": ai_status}


@app.post("/api/detect")
async def detect_deepfake(file: UploadFile = File(...)):
    """
    Receive uploaded video/image from frontend and proxy to AI service.
    Supports large files up to 500MB via streaming temp file.
    """
    ext = os.path.splitext(file.filename or "upload")[1].lower() or ".tmp"

    try:
        # Stream to temp file to support large uploads
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            chunk_size = 1024 * 1024  # 1MB chunks
            total_bytes = 0
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                tmp.write(chunk)
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_MB * 1024 * 1024:
                    os.unlink(tmp.name)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {MAX_UPLOAD_MB}MB"
                    )
            tmp_path = tmp.name

        file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        logger.info(f"Received: {file.filename} ({file_size_mb:.1f}MB)")

        # Forward to AI service with generous timeout for large videos
        timeout = max(120.0, file_size_mb * 2)  # 2 seconds per MB minimum
        async with httpx.AsyncClient(timeout=timeout) as client:
            with open(tmp_path, "rb") as f:
                files = {"file": (file.filename, f, file.content_type or "application/octet-stream")}
                response = await client.post(f"{AI_SERVICE_URL}/api/detect", files=files)

        os.unlink(tmp_path)

        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)

    except HTTPException:
        raise
    except httpx.TimeoutException:
        return JSONResponse(status_code=504, content={
            "error": "AI service timed out processing large video",
            "is_deepfake": False,
            "confidence": 0.0,
        })
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "is_deepfake": False,
            "confidence": 0.0,
        })
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=300,
    )
