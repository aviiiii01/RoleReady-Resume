# 🎯 RoleReady — AI-Powered Resume Tailor

RoleReady is an AI-powered resume automation tool that generates ATS-optimized, professionally tailored resumes in PDF format. Upload your existing resume or build one from scratch — the LLM agent pipeline extracts JD keywords, structures your details, and produces a publication-quality LaTeX resume designed to pass automated screening systems.

🔗 **Live App:** [https://role-ready-resume-one.vercel.app/](https://role-ready-resume-one.vercel.app/)

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | HTML5, CSS3 (dark theme + glassmorphism), Vanilla JavaScript |
| **Backend** | Python, FastAPI, Uvicorn |
| **AI / LLM** | Google Gemini 2.5 Flash, LangChain, LangGraph |
| **Resume Parsing** | PyMuPDF (fitz) |
| **Resume Output** | LaTeX (pdflatex), Pydantic structured output |
| **Deployment** | Vercel (frontend), Ngrok (backend tunnel) |

---

## 🚀 Setup

### Prerequisites

- **Python 3.10+**
- **pdflatex** (TeX Live) — for LaTeX → PDF compilation
- **Google Gemini API Key** — [Get one here](https://aistudio.google.com/api-keys/)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/RoleReady.git
cd RoleReady

# 2. Create & activate virtual environment
python -m venv myenv
source myenv/bin/activate        # macOS / Linux
myenv\Scripts\activate           # Windows

# 3. Install Python dependencies
pip install -r backend/requirements.txt

# 4. Install LaTeX (if not already installed)
# Ubuntu/Debian:
sudo apt install texlive-full
# macOS:
brew install --cask mactex

# 5. Set up your API key
echo "GOOGLE_API_KEY=your_gemini_api_key_here" > backend/.env
```

### Run the App

```bash
cd backend
uvicorn main:app --reload
```

Open **http://localhost:8000** in your browser.

---

## 📋 How to Use

### Mode 1 — Update Existing Resume

1. Select **"Update Existing Resume"** on the landing page
2. Upload your current resume PDF (drag & drop or browse)
3. Paste the full job description
4. Click **"Generate Tailored Resume"**
5. The AI will parse your resume, extract JD keywords, and generate a tailored PDF
6. PDF auto-downloads → ATS score with breakdown is displayed

### Mode 2 — Build from Scratch

1. Select **"Build from Scratch"** on the landing page
2. Fill in your details: name, email, phone, location, links
3. Add education, experience, projects, skills, and achievements
4. Paste the job description
5. Click **"Generate Resume from Scratch"**
6. The AI expands your entries into professional, quantified bullet points and generates the PDF
7. PDF auto-downloads → ATS score with breakdown is displayed

> **Tip:** The more detail you provide in experience and project descriptions, the better the generated resume will be.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
