import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from agent import process_jd

app = FastAPI(title="Resume Automation API")

# CORS middleware so the frontend can talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API route ───────────────────────────────────────────────────────
@app.post("/generate-resume")
async def generate_resume(
    jd_text: str = Form(...),
    resume_pdf: UploadFile = File(...)
):
    """
    Accepts a Job Description (text) and the user's existing resume (PDF).
    Returns a tailored resume PDF.
    """
    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    if not resume_pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    # Save uploaded PDF to a temp directory
    tmp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmp_dir, "uploaded_resume.pdf")

    try:
        with open(pdf_path, "wb") as f:
            content = await resume_pdf.read()
            f.write(content)

        # Run the agent pipeline
        result = process_jd(jd_text=jd_text.strip(), pdf_path=pdf_path)

        # The generated PDF is in the same directory as the uploaded PDF
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


# ── Serve static frontend ──────────────────────────────────────────
