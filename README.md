# 🎯 RoleReady — AI-Powered Resume Tailor

**RoleReady** is an intelligent resume automation tool that takes your existing resume (PDF) and a job description, then generates a professionally tailored resume optimized for that specific role. It leverages **LLM-powered agents** to parse your resume, extract JD keywords, and produce a publication-quality **LaTeX PDF** that maximizes your chances of passing ATS screening.

Check It Out here:-  https://role-ready-resume-one.vercel.app/
---

## ✨ Features

- **PDF Resume Upload** — Drag & drop or browse to upload your existing resume in PDF format
- **JD Keyword Extraction** — Automatically extracts critical keywords, skills, and technologies from the job description
- **AI Resume Parsing** — Intelligently parses your resume into structured fields: personal details, education, experience, projects, skills, and achievements
- **ATS-Optimized Tailoring** — Generates a resume that incorporates all JD keywords in the right sections, ensuring you pass automated screening
- **LaTeX PDF Output** — Produces clean, professional resumes using LaTeX with a modern template
- **Smart Retry Logic** — If LaTeX compilation fails, the system automatically retries with error-corrected code (up to 3 attempts)
- **Modern Dark-Themed UI** — Sleek frontend with drag & drop, progress indicators, and instant PDF download

---

## 🏗️ Architecture

```
RoleReady/
├── backend/
│   ├── agent.py          # LangGraph agent pipeline (4-node workflow)
│   ├── main.py           # FastAPI server with /generate-resume endpoint
│   ├── requirements.txt  # Python dependencies
│   ├── .env              # API keys (GOOGLE_API_KEY)
│   └── .gitignore
├── frontend/
│   ├── index.html        # Main UI — upload form, progress, download card
│   └── static/
│       ├── script.js     # Form submission, drag & drop, PDF blob download
│       └── style.css     # Dark theme with glassmorphism and animations
└── README.md
```

---

## 🔄 Agent Pipeline

The core of RoleReady is a **LangGraph** multi-step agent pipeline that processes your resume and the job description through four sequential nodes:

```
START → extract_pdf_text → extract_keywords → parse_resume_details → resume_gen → END
```

| Node | Description |
|------|-------------|
| **`extract_pdf_text`** | Reads the uploaded PDF with **PyMuPDF** and extracts all raw text content |
| **`extract_keywords`** | Sends the JD to **Gemini** LLM to extract all relevant keywords, skills, tools, and technologies |
| **`parse_resume_details`** | Uses **Gemini** with a Pydantic schema to parse the extracted text into structured fields — name, email, phone, education, experience, projects, skills, achievements |
| **`resume_gen`** | Generates a complete, ATS-optimized **LaTeX** resume using the candidate's real details and JD keywords, then compiles it to PDF via `pdflatex` |

### Tailoring Strategy

The `resume_gen` node applies intelligent tailoring:

- **Skills Section** — Ensures ALL JD keywords appear; organizes into categories (Languages, Frameworks, Tools, etc.)
- **Projects** — Displays tech stack in bold on the heading line (e.g., `ProjectName | Python, FastAPI, Redis`)
- **Experience** — Rephrases bullets using JD-specific terminology with action verbs and quantified achievements
- **Ordering** — Places the most JD-relevant items first in each section

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | HTML5, CSS3 (dark theme), Vanilla JavaScript |
| **Backend** | Python, FastAPI, Uvicorn |
| **AI/LLM** | Google Gemini 2.5 Flash, LangChain, LangGraph |
| **Resume Parsing** | PyMuPDF (fitz) |
| **Resume Generation** | LaTeX (pdflatex), Pydantic structured output |
| **Data Validation** | Pydantic BaseModel schemas |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **pdflatex** (TeX Live) — for LaTeX to PDF compilation
- **Google Gemini API Key** — [Get one here](https://ai.google.dev/)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/RoleReady.git
   cd RoleReady
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv myenv
   source myenv/bin/activate     # macOS/Linux
   myenv\Scripts\activate        # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Install LaTeX** (if not already installed)
   ```bash
   # Ubuntu/Debian
   sudo apt install texlive-full

   # macOS
   brew install --cask mactex
   ```

5. **Set up environment variables**

   Create a `.env` file in the `backend/` directory:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

### Running the Application

```bash
cd backend
uvicorn main:app --reload
```

Open **http://localhost:8000** in your browser.

---

## 📋 How to Use

1. **Upload your resume** — Drag & drop your existing resume PDF into the upload zone, or click "browse" to select it
2. **Paste the job description** — Copy the full JD text and paste it into the textarea
3. **Click "Generate Tailored Resume"** — The AI agent will:
   - Extract text from your PDF
   - Analyze the JD for keywords
   - Parse your resume details
   - Generate a tailored LaTeX resume
4. **Download** — The tailored resume PDF will automatically download once ready (typically 30–60 seconds)

---

## 📡 API Reference

### `POST /generate-resume`

Accepts multipart form data and returns a tailored resume PDF.

| Parameter | Type | Description |
|-----------|------|-------------|
| `jd_text` | `string` (Form) | The full job description text |
| `resume_pdf` | `file` (File) | The user's existing resume in PDF format |

**Success Response:** `200 OK` — Returns the generated PDF file (`application/pdf`)

**Error Responses:**
| Code | Detail |
|------|--------|
| `400` | Job description cannot be empty |
| `400` | Please upload a PDF file |
| `500` | Resume PDF generation failed / pdflatex errors |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
