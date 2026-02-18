import subprocess
import re
import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import fitz  # PyMuPDF


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)


# ── State ────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    jd_text: str
    pdf_path: str
    resume_text: str        # raw text extracted from the uploaded PDF
    keywords: list[str]
    parsed_details: str     # structured resume details (JSON-like string)
    resume: str             # final LaTeX code


# ── Structured output schemas ────────────────────────────────────────
class KeywordSchema(BaseModel):
    keywords: list[str]

Keyword_model = model.with_structured_output(KeywordSchema)


class ParsedResume(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str = Field(default="", description="Email address")
    phone: str = Field(default="", description="Phone number")
    location: str = Field(default="", description="City, State or Country")
    links: list[str] = Field(default_factory=list, description="LinkedIn, GitHub, portfolio URLs")
    education: list[str] = Field(description="Education entries, each as a single descriptive string")
    experience: list[str] = Field(description="Work experience entries, each as a single descriptive string")
    projects: list[str] = Field(default_factory=list, description="Project entries")
    skills: list[str] = Field(default_factory=list, description="Technical and soft skills")
    achievements: list[str] = Field(default_factory=list, description="Awards, certifications, achievements")

Resume_parser_model = model.with_structured_output(ParsedResume)


class LatexGen(BaseModel):
    resume: str = Field(
        description="Complete, valid LaTeX source code for a resume. "
                    "Must start with \\documentclass and include "
                    "\\begin{document} and \\end{document}. "
                    "No placeholders, no markdown fences, only raw LaTeX."
    )

Latex_model = model.with_structured_output(LatexGen)


# ── Node 1: Extract text from PDF ───────────────────────────────────
def extract_pdf_text(state: AgentState):
    pdf_path = state["pdf_path"]
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    print("=" * 50)
    print("PDF text extracted successfully (%d chars)" % len(text))
    print("=" * 50)
    return {"resume_text": text}


# ── Node 2: Extract keywords from JD ────────────────────────────────
def extract_keywords(state: AgentState):
    jd = state["jd_text"]
    prompt = f"""
    You are a JD Analyser Expert. 
    Analyze the given JD and extract the important keywords that are most relevant and important for a resume to match this JOB JD.
    \n\n
    JD: {jd}
    ANALYZE IT CAREFULLY AND EXTRACT ALL THE KEYWORDS.
    \n\n
    DO NOT PROVIDE ANYTHING EXCEPT THE KEYWORDS IN A LIST.
    """
    response = Keyword_model.invoke(prompt)
    print("=" * 50)
    print("Keywords: ", response.keywords)
    print("=" * 50)
    return {"keywords": response.keywords}


# ── Node 3: Parse resume details from extracted text ─────────────────
def parse_resume_details(state: AgentState):
    resume_text = state["resume_text"]
    prompt = f"""
      You are an expert Resume Parser. Given the raw text extracted from a PDF resume,
      parse it into structured fields. Extract every detail you can find.

      Raw resume text:
      \"\"\"
      {resume_text}
      \"\"\"

      Parse the above into: name, email, phone, location, links, education, experience,
      projects, skills, and achievements. Be thorough — do not miss any information.
      """
    response = Resume_parser_model.invoke(prompt)
    # Convert to a readable string for the next node
    details = f"""
      === CANDIDATE DETAILS ===
      Name: {response.name}
      Email: {response.email}
      Phone: {response.phone}
      Location: {response.location}
      Links: {', '.join(response.links) if response.links else 'N/A'}

      === EDUCATION ===
      {chr(10).join('- ' + e for e in response.education)}

      === EXPERIENCE ===
      {chr(10).join('- ' + e for e in response.experience)}

      === PROJECTS ===
      {chr(10).join('- ' + p for p in response.projects) if response.projects else 'N/A'}

      === SKILLS ===
      {', '.join(response.skills) if response.skills else 'N/A'}

      === ACHIEVEMENTS ===
      {chr(10).join('- ' + a for a in response.achievements) if response.achievements else 'N/A'}
      """
    print("=" * 50)
    print("Parsed resume details:")
    print(details[:500], "...")
    print("=" * 50)
    return {"parsed_details": details}


# ── LaTeX template (used as a style example for the LLM) ────────────
resume_latex = r"""
\documentclass[letterpaper,11pt]{article}

\usepackage{latexsym}
\usepackage[empty]{fullpage}
\usepackage{titlesec}
\usepackage{marvosym}
\usepackage[usenames,dvipsnames]{color}
\usepackage{verbatim}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\usepackage[english]{babel}
\usepackage{tabularx}

\usepackage{fontawesome5}
\usepackage[scale=0.90,lf]{FiraMono}

\definecolor{light-grey}{gray}{0.83}
\definecolor{dark-grey}{gray}{0.3}
\definecolor{text-grey}{gray}{.08}

\DeclareRobustCommand{\ebseries}{\fontseries{eb}\selectfont}
\DeclareTextFontCommand{\texteb}{\ebseries}

\usepackage{contour}
\usepackage[normalem]{ulem}
\renewcommand{\ULdepth}{1.8pt}
\contourlength{0.8pt}
\newcommand{\myuline}[1]{%
  \uline{\phantom{#1}}%
  \llap{\contour{white}{#1}}%
}

\usepackage{tgheros}
\renewcommand*\familydefault{\sfdefault}
\usepackage[T1]{fontenc}

\pagestyle{fancy}
\fancyhf{}
\fancyfoot{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

\addtolength{\oddsidemargin}{-0.5in}
\addtolength{\evensidemargin}{0in}
\addtolength{\textwidth}{1in}
\addtolength{\topmargin}{-.5in}
\addtolength{\textheight}{1.0in}

\urlstyle{same}
\raggedbottom
\raggedright
\setlength{\tabcolsep}{0in}

\titleformat {\section}{
    \bfseries \vspace{2pt} \raggedright \large
}{}{0em}{}[\color{light-grey} {\titlerule[2pt]} \vspace{-4pt}]

\newcommand{\resumeItem}[1]{
  \item\small{
    {#1 \vspace{-1pt}}
  }
}

\newcommand{\resumeSubheading}[4]{
  \vspace{-1pt}\item
    \begin{tabular*}{\textwidth}[t]{l@{\extracolsep{\fill}}r}
      \textbf{#1} & {\color{dark-grey}\small #2}\vspace{1pt}\\
      \textit{#3} & {\color{dark-grey} \small #4}\\
    \end{tabular*}\vspace{-4pt}
}

\newcommand{\resumeProjectHeading}[2]{
    \item
    \begin{tabular*}{\textwidth}{l@{\extracolsep{\fill}}r}
      #1 & {\color{dark-grey}} \\
    \end{tabular*}\vspace{-4pt}
}

\newcommand{\resumeSubItem}[1]{\resumeItem{#1}\vspace{-4pt}}
\renewcommand\labelitemii{$\vcenter{\hbox{\tiny$\bullet$}}$}

\newcommand{\resumeSubHeadingListStart}{\begin{itemize}[leftmargin=0in, label={}]}
\newcommand{\resumeSubHeadingListEnd}{\end{itemize}}
\newcommand{\resumeItemListStart}{\begin{itemize}}
\newcommand{\resumeItemListEnd}{\end{itemize}\vspace{0pt}}

\color{text-grey}

\begin{document}

\begin{center}
    \textbf{\Huge Harshibar} \\ \vspace{5pt}
    \small \faPhone* \texttt{555.555.5555} \hspace{1pt} $|$
    \hspace{1pt} \faEnvelope \hspace{2pt} \texttt{hello@email.com} \hspace{1pt} $|$
    \hspace{1pt} \faMapMarker* \hspace{2pt}\texttt{U.S. Citizen}
    \\ \vspace{-3pt}
\end{center}

\section{EXPERIENCE}
  \resumeSubHeadingListStart
    \resumeSubheading
      {Google Verily}{Aug. 2018 -- Sept. 2019}
      {Software Engineer}{San Francisco, CA}
      \resumeItemListStart
        \resumeItem{\textbf{Led front-end development} of a dashboard to process 50k blood samples}
        \resumeItem{Rebuilt a Quality Control product, \textbf{saving \$1M annually}}
      \resumeItemListEnd
  \resumeSubHeadingListEnd

\section{PROJECTS}
    \resumeSubHeadingListStart
      \resumeProjectHeading
          {\textbf{CommonIntern} $|$ \emph{Python, BeautifulSoup, Selenium}}{Sept. 2019 -- May 2020}
          \resumeItemListStart
            \resumeItem{Built a Python script to automatically apply to jobs using BeautifulSoup and Selenium}
          \resumeItemListEnd
    \resumeSubHeadingListEnd

\section{EDUCATION}
  \resumeSubHeadingListStart
    \resumeSubheading
      {Wellesley College}{Aug. 2014 -- May 2018}
      {Bachelor of Arts in Computer Science}{Wellesley, MA}
  \resumeSubHeadingListEnd

\section{SKILLS}
 \begin{itemize}[leftmargin=0in, label={}]
    \small{\item{
     \textbf{Languages} {: Python, JavaScript, HTML/CSS, SQL}\vspace{2pt} \\
     \textbf{Tools}     {: Git, Docker, Jira, Figma}
    }}
 \end{itemize}

\end{document}"""


# ── LaTeX sanitizer ──────────────────────────────────────────────────
def sanitize_latex(code: str) -> str:
    """
    Post-process LLM-generated LaTeX to fix common issues like
    unescaped & characters in text content.
    """
    lines = code.split('\n')
    result = []
    in_tabular = 0

    for line in lines:
        in_tabular += line.count('\\begin{tabular')
        in_tabular -= line.count('\\end{tabular')

        if in_tabular > 0:
            result.append(line)
        else:
            line = re.sub(r'(?<!\\)&', r'\\&', line)
            result.append(line)

    return '\n'.join(result)


# ── Node 4: Generate tailored LaTeX resume ───────────────────────────
def _build_latex_prompt(parsed_details, jd, keywords):
    return f"""
You are a LaTeX Resume Generator. Your ONLY job is to output a complete, compilable LaTeX resume
that is tailored to a specific Job Description using the candidate's REAL information.

STRICT RULES:
- Output ONLY raw LaTeX code. Nothing else.
- The output MUST start with \\documentclass
- The output MUST contain \\begin{{document}} and \\end{{document}}
- EVERY opening brace {{ MUST have a matching closing brace }}
- Do NOT invent or fabricate any information. Use ONLY the candidate's real details below.
- Do NOT wrap in markdown code fences (no ```)
- CRITICAL: Escape ALL special LaTeX characters in text content:
  - Use \\& instead of & (e.g. "C++ \\& Sockets", NOT "C++ & Sockets")
  - Use \\% instead of %
  - Use \\$ instead of $ (unless in math mode)
  - Use \\# instead of #

ATS SCREENING & TAILORING INSTRUCTIONS (VERY IMPORTANT):
1. SKILLS SECTION — MUST contain ALL keywords from the JD:
   - Include EVERY technology, tool, framework, language, and methodology mentioned in the JD.
   - If the candidate already knows a skill, keep it. If a JD skill is closely related to
     something the candidate knows, ADD it (e.g., if candidate knows React and JD mentions Next.js, include both).
   - Organize skills into categories like: Languages, Frameworks, Tools, Databases, Cloud, etc.
   - This section is critical for ATS screening — missing keywords here means rejection.

2. PROJECTS FORMAT — Tech skills on the heading line:
   - Each project MUST show its tech stack in bold+italic on the SAME line as the project name.
   - Format: \\resumeProjectHeading{{\\textbf{{Project Name}} $|$ \\emph{{Tech1, Tech2, Tech3}}}}{{Date}}
   - List the ACTUAL technologies used in each project.
   - In project bullet points, naturally mention JD-relevant technologies where applicable.

3. EXPERIENCE BULLETS — Maximize keyword coverage:
   - Rephrase and reorder the candidate's experience bullets to emphasize skills matching the JD.
   - Use JD-specific terminology where the candidate has equivalent experience.
   - Use action verbs and quantified achievements (numbers, percentages, scale).
   - Weave JD keywords naturally into bullet points (e.g., if JD says "microservices",
     mention it in relevant experience bullets).

4. GENERAL:
   - Place the most JD-relevant skills, experiences, and projects prominently (first in each section).
   - Keep ALL of the candidate's real information (name, education, contact, etc.).
   - The resume MUST feel like a strong match when an ATS or recruiter scans it against the JD.

=== CANDIDATE'S REAL INFORMATION ===
{parsed_details}

=== JOB DESCRIPTION ===
{jd}

=== KEYWORDS TO EMPHASIZE (ALL of these MUST appear somewhere in the resume) ===
{keywords}

=== EXAMPLE LATEX TEMPLATE (follow this style/structure EXACTLY) ===
{resume_latex}

OUTPUT ONLY THE COMPLETE LATEX CODE STARTING WITH \\documentclass. NO OTHER TEXT.
"""


def   _generate_and_compile(prompt, output_dir):
    """Generate LaTeX from LLM and compile to PDF. Returns (success, latex_code, error_msg)."""
    response = Latex_model.invoke(prompt)
    latex_code = response.resume

    # Safety check
    if "\\begin{document}" not in latex_code:
        return False, latex_code, "Missing \\begin{document}"

    # Strip markdown fences just in case
    if "```" in latex_code:
        lines = latex_code.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        latex_code = "\n".join(lines)

    # Sanitize LaTeX
    latex_code = sanitize_latex(latex_code)

    tex_path = os.path.join(output_dir, "resume.tex")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_code)

    for _ in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode",
             "-output-directory", output_dir, tex_path],
            capture_output=True, text=True
        )

    if result.returncode != 0:
        # Extract the key error line from the log
        error_lines = [l for l in result.stdout.splitlines() if l.startswith("!")]
        error_msg = "; ".join(error_lines[:3]) if error_lines else "Unknown LaTeX error"
        return False, latex_code, error_msg

    return True, latex_code, None


def resume_gen(state: AgentState):
    jd = state["jd_text"]
    keywords = state["keywords"]
    parsed_details = state["parsed_details"]
    output_dir = os.path.dirname(state["pdf_path"]) or "."
    pdf_path = os.path.join(output_dir, "resume.pdf")

    MAX_RETRIES = 3
    prompt = _build_latex_prompt(parsed_details, jd, keywords)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[resume_gen] Attempt {attempt}/{MAX_RETRIES}")
        success, latex_code, error_msg = _generate_and_compile(prompt, output_dir)

        if success:
            print("=" * 50)
            print(f"PDF generated successfully at {pdf_path}!")
            print("=" * 50)
            return {"resume": latex_code}

        print(f"[resume_gen] Attempt {attempt} FAILED: {error_msg}")

        if attempt < MAX_RETRIES:
            # Ask the LLM to fix the error on the next attempt
            prompt = f"""
              The following LaTeX code failed to compile with this error:
              {error_msg}

              Fix the LaTeX code below so it compiles successfully. Make sure every brace is matched,
              all commands are valid, and the document is complete from \\documentclass to \\end{{document}}.

              Broken code:
              \"\"\"
              {latex_code}
              \"\"\"

              OUTPUT ONLY THE FIXED, COMPLETE LATEX CODE. NO OTHER TEXT.
              """

    raise RuntimeError(
        f"pdflatex failed after {MAX_RETRIES} attempts. Last error: {error_msg}"
    )


# ── Build the graph ──────────────────────────────────────────────────
workflow = StateGraph(AgentState)

workflow.add_node("extract_pdf_text", extract_pdf_text)
workflow.add_node("extract_keywords", extract_keywords)
workflow.add_node("parse_resume_details", parse_resume_details)
workflow.add_node("resume_gen", resume_gen)

# Flow: START → extract_pdf_text → extract_keywords → parse_resume_details → resume_gen → END
workflow.add_edge(START, "extract_pdf_text")
workflow.add_edge("extract_pdf_text", "extract_keywords")
workflow.add_edge("extract_keywords", "parse_resume_details")
workflow.add_edge("parse_resume_details", "resume_gen")
workflow.add_edge("resume_gen", END)

agent_app = workflow.compile()


# ── Public API ───────────────────────────────────────────────────────
def process_jd(jd_text: str, pdf_path: str) -> dict:
    """
    Run the full agent pipeline.
    Args:
        jd_text:  The job description text.
        pdf_path: Absolute path to the user's uploaded resume PDF.
    Returns:
        dict with keys: keywords, parsed_details, resume (LaTeX code)
    """
    result = agent_app.invoke({
        "jd_text": jd_text,
        "pdf_path": pdf_path,
    })
    return result
