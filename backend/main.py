import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from agent import process_jd, calculate_ats_score

app = FastAPI(title="Resume Automation API")

# CORS middleware so the frontend can talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Generate Resume ─────────────────────────────────────────────────
@app.post("/generate-resume")
async def generate_resume(
    jd_text: str = Form(...),
    mode: str = Form("update"),
    resume_pdf: UploadFile = File(None),
    candidate_details: str = Form(None),
):
    """
    Accepts a Job Description (text) and optionally the user's existing resume (PDF).
    mode='update' requires a PDF; mode='scratch' sends candidate_details instead.
    Returns a tailored resume PDF.
    """
    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    if mode not in ("scratch", "update"):
        raise HTTPException(status_code=400, detail="Mode must be 'scratch' or 'update'.")

    tmp_dir = tempfile.mkdtemp()
    pdf_path = None

    if mode == "update":
        if not resume_pdf or not resume_pdf.filename:
            raise HTTPException(status_code=400, detail="Please upload a resume PDF for update mode.")
        if not resume_pdf.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a PDF file.")

        pdf_path = os.path.join(tmp_dir, "uploaded_resume.pdf")
        with open(pdf_path, "wb") as f:
            content = await resume_pdf.read()
            f.write(content)
    else:
        # Scratch mode — create a dummy path so the output goes into tmp_dir
        pdf_path = os.path.join(tmp_dir, "placeholder.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"")  # empty file, won't be read

    try:
        result = process_jd(
            jd_text=jd_text.strip(),
            pdf_path=pdf_path,
            mode=mode,
            candidate_details=candidate_details or "",
        )

        generated_pdf = os.path.join(tmp_dir, "resume.pdf")

        if not os.path.exists(generated_pdf):
            raise HTTPException(status_code=500, detail="Resume PDF generation failed.")

        return FileResponse(
            path=generated_pdf,
            media_type="application/pdf",
            filename="tailored_resume.pdf",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ATS Score ────────────────────────────────────────────────────────
@app.post("/ats-score")
async def ats_score(
    jd_text: str = Form(...),
    resume_pdf: UploadFile = File(...),
):
    """
    Accepts a generated resume PDF and JD text.
    Returns an ATS compatibility score with breakdown and suggestions.
    """
    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    if not resume_pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    tmp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmp_dir, "resume_to_score.pdf")

    try:
        with open(pdf_path, "wb") as f:
            content = await resume_pdf.read()
            f.write(content)

        score_result = calculate_ats_score(resume_pdf_path=pdf_path, jd_text=jd_text.strip())
        return JSONResponse(content=score_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ATS scoring failed: {str(e)}")


# ── Serve static frontend ──────────────────────────────────────────
