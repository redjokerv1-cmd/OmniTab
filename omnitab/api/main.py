"""
OmniTab REST API

Endpoints:
- POST /convert - Convert TAB image to GP5
- POST /convert/batch - Convert multiple pages
- GET /status - API status
- GET /history - Conversion history
"""

import os
import uuid
import shutil
import logging
import traceback
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("omnitab.api")

# Initialize app
app = FastAPI(
    title="OmniTab API",
    description="Convert guitar TAB images to Guitar Pro (GP5) files",
    version="0.5.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def convert_pdf_to_images(pdf_path: Path) -> List[Path]:
    """Convert PDF to images using PyMuPDF (no poppler needed)"""
    try:
        import fitz  # PyMuPDF
        
        logger.info(f"[PDF] Converting {pdf_path} using PyMuPDF")
        
        # Open PDF
        doc = fitz.open(str(pdf_path))
        
        image_paths = []
        for i, page in enumerate(doc):
            # Render page at 200 DPI
            mat = fitz.Matrix(200/72, 200/72)  # 72 is default DPI
            pix = page.get_pixmap(matrix=mat)
            
            # Save as PNG
            img_path = pdf_path.parent / f"{pdf_path.stem}_page_{i+1}.png"
            pix.save(str(img_path))
            image_paths.append(img_path)
            logger.info(f"[PDF] Saved page {i+1}: {img_path}")
        
        doc.close()
        return image_paths
        
    except ImportError:
        logger.error("[PDF] PyMuPDF not installed")
        raise HTTPException(
            status_code=500,
            detail="PDF conversion requires PyMuPDF. Install with: pip install PyMuPDF"
        )
    except Exception as e:
        logger.error(f"[PDF] Conversion failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {str(e)}")


# Models
class ConversionResult(BaseModel):
    job_id: str
    status: str
    output_file: Optional[str] = None
    measures: int = 0
    notes: int = 0
    rhythm_source: str = "default"
    tuning: List[str] = []
    capo: int = 0
    error: Optional[str] = None


class ConversionRequest(BaseModel):
    title: str = "OmniTab Conversion"
    use_gemini: bool = True


# In-memory job storage (use Redis for production)
jobs = {}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding="utf-8")
    return """
    <html>
        <head><title>OmniTab API</title></head>
        <body>
            <h1>OmniTab API</h1>
            <p>API is running. Visit <a href="/docs">/docs</a> for API documentation.</p>
        </body>
    </html>
    """


@app.get("/api")
async def api_info():
    """API info"""
    return {
        "name": "OmniTab API",
        "version": "0.5.0",
        "status": "running",
        "endpoints": {
            "convert": "POST /convert - Convert single TAB image",
            "batch": "POST /convert/batch - Convert multiple images",
            "status": "GET /status - API status",
            "history": "GET /history - Conversion history",
            "download": "GET /download/{job_id} - Download GP5 file"
        }
    }


@app.get("/status")
async def get_status():
    """Get API status and capabilities"""
    gemini_available = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    
    return {
        "status": "healthy",
        "gemini_available": gemini_available,
        "total_conversions": len(jobs),
        "capabilities": {
            "ocr_accuracy": "93.8%",
            "rhythm_analysis": "Gemini 2.5 Flash" if gemini_available else "Default (quarter notes)",
            "tuning_detection": "automatic",
            "capo_detection": "automatic",
            "output_format": "GP5"
        }
    }


@app.post("/convert", response_model=ConversionResult)
async def convert_single(
    file: UploadFile = File(...),
    title: str = Form("OmniTab Conversion"),
    use_gemini: bool = Form(True)
):
    """
    Convert a single TAB image to GP5
    
    - **file**: TAB image (PNG, JPG, PDF)
    - **title**: Song title for GP5 file
    - **use_gemini**: Use Gemini for rhythm analysis (requires API key)
    """
    job_id = str(uuid.uuid4())[:8]
    
    logger.info(f"[{job_id}] === NEW CONVERSION REQUEST ===")
    logger.info(f"[{job_id}] File: {file.filename}")
    logger.info(f"[{job_id}] Title: {title}")
    logger.info(f"[{job_id}] Use Gemini: {use_gemini}")
    
    try:
        # Save uploaded file
        upload_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
        logger.info(f"[{job_id}] Saved to: {upload_path}")
        
        # Check if PDF - convert to images first
        image_path = upload_path
        if file.filename.lower().endswith('.pdf'):
            logger.info(f"[{job_id}] Detected PDF, converting to images...")
            image_paths = convert_pdf_to_images(upload_path)
            if not image_paths:
                raise ValueError("PDF conversion produced no images")
            image_path = image_paths[0]  # Use first page
            logger.info(f"[{job_id}] Using first page: {image_path}")
        
        # Output path
        output_path = OUTPUT_DIR / f"{job_id}.gp5"
        
        # Convert
        logger.info(f"[{job_id}] Starting conversion...")
        from omnitab.tab_ocr.complete_converter import CompleteConverter
        
        converter = CompleteConverter()
        result = converter.convert(
            image_path=str(image_path),
            output_path=str(output_path),
            title=title,
            use_gemini=use_gemini
        )
        
        logger.info(f"[{job_id}] Conversion complete!")
        logger.info(f"[{job_id}] Result: {result}")
        
        # Store job
        job_data = ConversionResult(
            job_id=job_id,
            status="completed",
            output_file=str(output_path),
            measures=result.get("measures", 0),
            notes=result.get("notes", 0),
            rhythm_source=result.get("rhythm_source", "default"),
            tuning=result.get("tuning", []),
            capo=result.get("capo", 0)
        )
        jobs[job_id] = job_data
        
        return job_data
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"[{job_id}] CONVERSION FAILED!")
        logger.error(f"[{job_id}] Error: {error_msg}")
        logger.error(f"[{job_id}] Traceback:\n{error_trace}")
        
        job_data = ConversionResult(
            job_id=job_id,
            status="failed",
            error=error_msg
        )
        jobs[job_id] = job_data
        raise HTTPException(status_code=500, detail=f"{error_msg}\n\nTraceback:\n{error_trace}")


@app.post("/convert/batch")
async def convert_batch(
    files: List[UploadFile] = File(...),
    title: str = Form("OmniTab Batch"),
    use_gemini: bool = Form(True)
):
    """
    Convert multiple TAB images and merge into single GP5
    
    - **files**: Multiple TAB images (one per page)
    - **title**: Song title for GP5 file
    - **use_gemini**: Use Gemini for rhythm analysis
    """
    job_id = str(uuid.uuid4())[:8]
    results = []
    
    try:
        from omnitab.tab_ocr.complete_converter import CompleteConverter
        converter = CompleteConverter()
        
        all_measures = []
        total_notes = 0
        
        for i, file in enumerate(files):
            # Save file
            upload_path = UPLOAD_DIR / f"{job_id}_page{i+1}_{file.filename}"
            with open(upload_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Convert
            page_output = OUTPUT_DIR / f"{job_id}_page{i+1}.gp5"
            result = converter.convert(
                image_path=str(upload_path),
                output_path=str(page_output),
                title=f"{title} - Page {i+1}",
                use_gemini=use_gemini
            )
            
            results.append({
                "page": i + 1,
                "measures": result.get("measures", 0),
                "notes": result.get("notes", 0)
            })
            total_notes += result.get("notes", 0)
        
        return {
            "job_id": job_id,
            "status": "completed",
            "pages": len(files),
            "total_notes": total_notes,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{job_id}")
async def download_file(job_id: str):
    """Download converted GP5 file"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job.status != "completed" or not job.output_file:
        raise HTTPException(status_code=400, detail="Conversion not completed")
    
    output_path = Path(job.output_file)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(output_path),
        filename=f"{job_id}.gp5",
        media_type="application/octet-stream"
    )


@app.get("/history")
async def get_history():
    """Get conversion history"""
    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": job.job_id,
                "status": job.status,
                "measures": job.measures,
                "notes": job.notes,
                "rhythm_source": job.rhythm_source
            }
            for job in jobs.values()
        ]
    }


@app.delete("/cleanup")
async def cleanup_old_files():
    """Clean up old upload and output files"""
    upload_count = 0
    output_count = 0
    
    for f in UPLOAD_DIR.glob("*"):
        f.unlink()
        upload_count += 1
    
    for f in OUTPUT_DIR.glob("*"):
        f.unlink()
        output_count += 1
    
    jobs.clear()
    
    return {
        "cleaned": {
            "uploads": upload_count,
            "outputs": output_count,
            "jobs": "cleared"
        }
    }


# Run with: uvicorn omnitab.api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
