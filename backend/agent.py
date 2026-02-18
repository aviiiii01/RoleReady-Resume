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
    mode: str               # 'scratch' or 'update'
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
You are an expert JD Keyword Analyst. Analyze the following Job Description and extract
ALL important keywords that a resume MUST contain to pass ATS screening.

Extract keywords from these categories:
1. Programming languages (e.g., Python, Java, JavaScript, C++)
2. Frameworks & Libraries (e.g., React, FastAPI, Django, Spring Boot)
3. Tools & Platforms (e.g., Docker, Kubernetes, Git, Jenkins, Jira)
4. Databases (e.g., PostgreSQL, MongoDB, Redis, MySQL)
5. Cloud & Infrastructure (e.g., AWS, GCP, Azure, Terraform)
6. Methodologies (e.g., Agile, Scrum, CI/CD, TDD, Microservices)
7. Soft skills & responsibilities (e.g., Team Leadership, Cross-functional, Stakeholder Management)
8. Domain-specific terms (e.g., Machine Learning, REST API, GraphQL, Data Pipeline)

JD:
{jd}

Be EXHAUSTIVE — missing even one important keyword can cause ATS rejection.
Return ALL keywords as a flat list. Include both the full term and common abbreviations (e.g., both "Amazon Web Services" and "AWS").
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
      \parbox{0.78\textwidth}{#1} & {\color{dark-grey}\small #2} \\
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
def _build_latex_prompt(parsed_details, jd, keywords, mode="update"):
    candidate_section = ""
    tailoring_note = ""

    if mode == "scratch" and not parsed_details:
        # No details at all — pure template
        candidate_section = "(No existing resume provided — create a strong template resume structure with placeholder sections that match the JD.)"
        tailoring_note = """TEMPLATE MODE:
   - Generate a COMPLETE, professional resume that a strong candidate targeting this JD would have.
   - Use [Your Name], [your.email@example.com], [+1-XXX-XXX-XXXX], [City, State] as placeholders for personal info.
   - Create 2-3 realistic work experiences with JD-relevant roles, companies, and dates.
   - Create 2-3 technical projects using the JD's tech stack.
   - Each bullet point MUST follow the XYZ formula: "Accomplished [X] as measured by [Y], by doing [Z]"
   - Include realistic quantified achievements (%, $, scale numbers) in every bullet."""
    elif mode == "scratch" and parsed_details:
        # User provided details via the form — use their real info but professionally expand it
        candidate_section = parsed_details
        tailoring_note = """SCRATCH MODE WITH CANDIDATE DETAILS:
   - Use the candidate's REAL name, email, phone, location, and links exactly as provided.
   - Use the candidate's REAL education details exactly as provided.
   - For experience & projects: the candidate provided brief descriptions. You MUST:
     * Expand each entry into 3-4 professional, ATS-optimized bullet points.
     * Rewrite bullets using the XYZ formula: "Accomplished [X] as measured by [Y], by doing [Z]"
     * Add realistic quantified metrics (e.g., "reduced latency by 35%", "serving 10K+ users", "processed 1M+ records")
     * Weave in JD-relevant keywords and technologies naturally into each bullet.
     * Use strong action verbs: Engineered, Architected, Spearheaded, Optimized, Implemented, Automated, etc.
   - If the candidate has gaps (few projects, limited experience), create ADDITIONAL bullets that
     expand on the given details — but do NOT invent entirely new experiences or projects.
   - The Skills section MUST include ALL the candidate's listed skills PLUS any JD keywords the candidate is likely to know.
   - Achievements/certifications should be included as-is."""
    else:
        # Update mode — use parsed resume data, only tailor the presentation
        candidate_section = parsed_details
        tailoring_note = """UPDATE MODE:
   - Do NOT invent or fabricate any new experiences, projects, or companies. Use ONLY the candidate's real details.
   - Keep ALL of the candidate's real information (name, education, contact, etc.).
   - Rephrase and reorder experience bullets to emphasize skills matching the JD keywords.
   - Rewrite bullets using the XYZ formula: "Accomplished [X] as measured by [Y], by doing [Z]"
   - Transform weak bullets into strong ones:
     * BAD:  "Worked on backend APIs"
     * GOOD: "Engineered 15+ RESTful APIs using FastAPI, reducing average response time by 40\\% and serving 50K+ daily requests"
   - If the candidate already has quantified metrics, keep them. If not, add realistic estimates.
   - Reorder sections and items so the most JD-relevant content appears first."""

    return f"""
You are an ELITE Resume Engineer and ATS Optimization Specialist. Your job is to produce a
PERFECT, high-scoring, ATS-friendly LaTeX resume tailored to a specific Job Description.
The resume MUST score 85+ on any ATS system.

═══════════════════════════════════════════════════════════════════
LATEX COMPILATION RULES (NON-NEGOTIABLE):
═══════════════════════════════════════════════════════════════════
- Output ONLY raw LaTeX code. Nothing else — no explanations, no markdown.
- The output MUST start with \\documentclass
- The output MUST contain \\begin{{document}} and \\end{{document}}
- EVERY opening brace {{ MUST have a matching closing brace }}
- Do NOT wrap in markdown code fences (no ```)
- CRITICAL: Escape ALL special LaTeX characters in text content:
  - Use \\& instead of & (e.g. "C++ \\& Sockets", NOT "C++ & Sockets")
  - Use \\% instead of %
  - Use \\$ instead of $ (unless in math mode)
  - Use \\# instead of #
  - Use \\_ instead of _ in text (URLs, company names, etc.)

═══════════════════════════════════════════════════════════════════
MODE & TAILORING INSTRUCTIONS:
═══════════════════════════════════════════════════════════════════
{tailoring_note}

═══════════════════════════════════════════════════════════════════
ATS OPTIMIZATION RULES (CRITICAL — FOLLOW ALL):
═══════════════════════════════════════════════════════════════════

1. SECTION ORDER (top to bottom):
   - Header (Name, Phone, Email, Location, Links)
   - Education (if recent grad) OR Experience (if experienced)
   - Experience / Education (the other one)
   - Projects (2-3 most relevant)
   - Skills (MUST be last or second-to-last)

2. SKILLS SECTION — THE ATS KEYWORD GOLDMINE:
   - Include EVERY technology, tool, framework, language, and methodology from the JD.
   - Organize into clear categories: Languages | Frameworks | Tools | Databases | Cloud | Methodologies
   - If the JD mentions "React", "Node.js", "AWS", "Docker" — ALL must appear here.
   - This section alone can determine pass/fail for ATS screening.
   - Include 15-25 skills minimum, organized into 3-5 categories.

3. EXPERIENCE BULLETS — PROFESSIONAL & QUANTIFIED:
   - Every bullet MUST follow the XYZ formula: "Accomplished [X] as measured by [Y], by doing [Z]"
   - Every bullet MUST start with a STRONG action verb: Engineered, Architected, Spearheaded,
     Optimized, Implemented, Automated, Designed, Developed, Led, Deployed, Integrated, Built,
     Launched, Streamlined, Reduced, Increased, Migrated, Orchestrated, Scaled
   - NEVER use: "Responsible for", "Worked on", "Helped with", "Assisted in"
   - Each bullet MUST include at least ONE of: percentage, dollar amount, user count, time saved, scale metric
   - Examples of GREAT bullets:
     * "Architected a microservices platform using FastAPI \\& Kubernetes, reducing deployment time by 60\\% and supporting 100K+ concurrent users"
     * "Engineered a real-time data pipeline with Apache Kafka \\& PySpark, processing 2M+ events/day with 99.9\\% uptime"
     * "Led migration of monolithic application to React \\& Next.js, improving page load speed by 45\\% and increasing user engagement by 30\\%"
   - 3-5 bullets per experience entry. Each bullet should be 1-2 lines max.

4. PROJECTS SECTION — TECH STACK ON HEADING LINE:
   - Format: \\resumeProjectHeading{{\\textbf{{Project Name}} $|$ \\emph{{Tech1, Tech2, Tech3}}}}{{Date}}
   - CRITICAL: List only 3-5 KEY technologies on the heading line. Do NOT list every tech — pick the most
     JD-relevant ones. The heading MUST fit on one line. Additional techs can go in the bullets.
   - Each project gets 2-3 strong bullets with quantified results.
   - Tech stacks MUST align with JD keywords.
   - Project bullets should demonstrate hands-on building, not just describing.

5. EDUCATION SECTION:
   - Include degree, university, dates, GPA (if > 3.5 / 8.0), relevant coursework.
   - Add relevant coursework ONLY if it matches JD requirements.

6. HEADER — CLEAN & COMPLETE:
   - Name prominently displayed.
   - Phone, Email, Location on one line with icons.
   - LinkedIn, GitHub, Portfolio links (if available) on the same or next line.
   - Use \\faPhone*, \\faEnvelope, \\faMapMarker*, \\faLinkedin, \\faGithub icons.

7. ACHIEVEMENTS & CERTIFICATIONS SECTION:
   - Use simple \\resumeItem bullet points. Do NOT use \\resumeSubheading.
   - Format: \\resumeItem{{\\textbf{{Certificate/Award Name}} -- Issuing Organization, Date}}
   - Keep each entry to a SINGLE bullet point. No sub-bullets or descriptions unless very brief.
   - Example:
     \\resumeItem{{\\textbf{{AWS Certified Cloud Practitioner}} -- Amazon Web Services, 2024}}
     \\resumeItem{{\\textbf{{Problem Solving Certificate}} -- HackerRank, Nov 2024}}

8. FORMATTING — MAXIMAL CONTENT DENSITY:
   - The resume should be EXACTLY 1 page. Use every inch of space.
   - Tight margins (already set in template). Do NOT add extra vertical spacing.
   - No wasted space — fill the page with relevant, impactful content.
   - Use consistent date formatting: "Mon. YYYY -- Mon. YYYY" or "Mon YYYY -- Present".

═══════════════════════════════════════════════════════════════════
CANDIDATE'S INFORMATION:
═══════════════════════════════════════════════════════════════════
{candidate_section}

═══════════════════════════════════════════════════════════════════
JOB DESCRIPTION:
═══════════════════════════════════════════════════════════════════
{jd}

═══════════════════════════════════════════════════════════════════
KEYWORDS TO EMPHASIZE (ALL of these MUST appear in the resume):
═══════════════════════════════════════════════════════════════════
{keywords}

═══════════════════════════════════════════════════════════════════
LATEX TEMPLATE (follow this style/structure EXACTLY):
═══════════════════════════════════════════════════════════════════
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
    mode = state["mode"]
    pdf_path_val = state.get("pdf_path", "")
    output_dir = os.path.dirname(pdf_path_val) if pdf_path_val else tempfile.mkdtemp()
    pdf_path = os.path.join(output_dir, "resume.pdf")

    MAX_RETRIES = 3
    prompt = _build_latex_prompt(parsed_details, jd, keywords, mode=mode)

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


import tempfile

# ── Routing ──────────────────────────────────────────────────────────
def _route_after_start(state: AgentState):
    """Route based on mode: skip PDF extraction for scratch mode."""
    if state.get("mode") == "scratch":
        return "extract_keywords"
    return "extract_pdf_text"


def _route_after_keywords(state: AgentState):
    """Route based on mode: skip resume parsing for scratch mode."""
    if state.get("mode") == "scratch":
        return "resume_gen"
    return "parse_resume_details"


# ── Build the graph ──────────────────────────────────────────────────
workflow = StateGraph(AgentState)

workflow.add_node("extract_pdf_text", extract_pdf_text)
workflow.add_node("extract_keywords", extract_keywords)
workflow.add_node("parse_resume_details", parse_resume_details)
workflow.add_node("resume_gen", resume_gen)

# Conditional start: scratch skips PDF extraction
workflow.add_conditional_edges(START, _route_after_start, {
    "extract_pdf_text": "extract_pdf_text",
    "extract_keywords": "extract_keywords",
})
workflow.add_edge("extract_pdf_text", "extract_keywords")

# Conditional after keywords: scratch skips resume parsing
workflow.add_conditional_edges("extract_keywords", _route_after_keywords, {
    "parse_resume_details": "parse_resume_details",
    "resume_gen": "resume_gen",
})
workflow.add_edge("parse_resume_details", "resume_gen")
workflow.add_edge("resume_gen", END)

agent_app = workflow.compile()


# ── ATS Score ────────────────────────────────────────────────────────
class ATSScoreResult(BaseModel):
    overall_score: int = Field(description="Overall ATS compatibility score from 0 to 100")
    keyword_match: int = Field(description="Keyword match score 0-100")
    skills_coverage: int = Field(description="Skills coverage score 0-100")
    experience_relevance: int = Field(description="Experience relevance score 0-100")
    formatting_score: int = Field(description="Formatting and ATS readability score 0-100")
    suggestions: list[str] = Field(description="3-5 specific improvement suggestions")

ATS_model = model.with_structured_output(ATSScoreResult)


def calculate_ats_score(resume_pdf_path: str, jd_text: str) -> dict:
    """
    Score a generated resume PDF against a job description.
    Returns a dict with overall_score, breakdown, and suggestions.
    """
    # Extract text from the generated resume
    resume_text = ""
    with fitz.open(resume_pdf_path) as doc:
        for page in doc:
            resume_text += page.get_text()

    prompt = f"""
You are an advanced Applicant Tracking System (ATS) scoring engine similar to Resume Worded.
Analyze the provided resume against the job description and generate scores.

=== RESUME ===
{resume_text}

=== JOB DESCRIPTION ===
{jd_text}

Score the resume on these dimensions (each 0-100):

1. **keyword_match** (weight 40%): What percentage of important JD keywords
   (skills, tools, technologies, role-specific terms) appear in the resume?

2. **skills_coverage** (weight 25%): How well does the Skills section cover
   the required technologies, frameworks, and tools from the JD?

3. **experience_relevance** (weight 25%): How relevant are the experience and
   project bullets to the JD? Do they use strong action verbs and quantified metrics?

4. **formatting_score** (weight 10%): Is the resume well-structured, ATS-parseable,
   uses bullet points, and follows professional formatting?

Then compute the **overall_score** as:
  overall_score = 0.40 * keyword_match + 0.25 * skills_coverage + 0.25 * experience_relevance + 0.10 * formatting_score

Also provide 3-5 specific, actionable **suggestions** for improvement.

Be strict and realistic. Do not inflate scores.
If major required skills are missing, reduce keyword_match and skills_coverage significantly.
"""
    result = ATS_model.invoke(prompt)
    return {
        "overall_score": result.overall_score,
        "breakdown": {
            "keyword_match": result.keyword_match,
            "skills_coverage": result.skills_coverage,
            "experience_relevance": result.experience_relevance,
            "formatting_score": result.formatting_score,
        },
        "suggestions": result.suggestions,
    }


# ── Public API ───────────────────────────────────────────────────────
def process_jd(jd_text: str, pdf_path: str = None, mode: str = "update", candidate_details: str = "") -> dict:
    """
    Run the full agent pipeline.
    Args:
        jd_text:           The job description text.
        pdf_path:          Absolute path to the user's uploaded resume PDF (None for scratch mode).
        mode:              'scratch' or 'update'
        candidate_details: Structured candidate info from the scratch form (only used in scratch mode).
    Returns:
        dict with keys: keywords, parsed_details, resume (LaTeX code)
    """
    input_state = {
        "jd_text": jd_text,
        "mode": mode,
        "pdf_path": pdf_path or "",
    }
    # In scratch mode, inject the form data as pre-parsed details
    if mode == "scratch" and candidate_details:
        input_state["parsed_details"] = candidate_details
    print(input_state)
    result = agent_app.invoke(input_state)
    return result

