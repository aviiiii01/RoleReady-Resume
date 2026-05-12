import warnings
warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change",
)

import subprocess
import re
import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import fitz  # PyMuPDF


load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


def _create_model(counter: int):
    """Create a ChatGoogleGenerativeAI model with the API key for the given counter (1-10)."""
    api_key = os.getenv(f"GEMINI_API_KEY{counter}")
    if not api_key:
        raise RuntimeError(f"GEMINI_API_KEY{counter} not found in environment variables.")
    print(f"[Model Init] Using GEMINI_API_KEY{counter}")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, api_key=api_key)


# ── State ────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    jd_text: str
    mode: str               # 'scratch' or 'update'
    pdf_path: str
    resume_text: str        # raw text extracted from the uploaded PDF
    keywords: list[str]
    parsed_details: str     # structured resume details (JSON-like string)
    resume: str             # final LaTeX code
    api_counter: int        # which API key (1-10) to use


# ── Structured output schemas ────────────────────────────────────────
class KeywordSchema(BaseModel):
    role_title: str = Field(description="The exact job title from the JD (e.g., 'Senior Backend Engineer', 'Content Writer')")
    industry_domain: str = Field(description="The industry or domain (e.g., 'FinTech', 'Dating App', 'Freelance Platform', 'Healthcare SaaS')")
    seniority_level: str = Field(description="Seniority level: 'Intern', 'Junior', 'Mid', 'Senior', 'Lead', 'Staff', 'Principal', 'Manager'")
    core_responsibilities: list[str] = Field(description="3-5 core responsibilities distilled from the JD, each as a concise phrase")
    hard_skills: list[str] = Field(description="Technical skills, tools, languages, frameworks, platforms explicitly mentioned or strongly implied by the JD")
    soft_skills: list[str] = Field(description="Soft skills, work style traits, and interpersonal abilities the JD asks for")
    domain_keywords: list[str] = Field(description="Industry-specific terminology, concepts, and buzzwords relevant to this role")
    keywords: list[str] = Field(description="Final flat list of ALL ATS-critical keywords combining hard skills, soft skills, and domain terms")


class ParsedResume(BaseModel):
    name: str = Field(description="Full name of the candidate exactly as it appears on the resume")
    email: str = Field(default="", description="Email address of the candidate")
    phone: str = Field(default="", description="Phone number including country code if present")
    location: str = Field(default="", description="City, State/Province, Country as listed on the resume")
    links: list[str] = Field(
        default_factory=list,
        description="ALL profile/portfolio URLs found on the resume — LinkedIn, GitHub, personal website, Behance, Dribbble, Medium, LeetCode, etc. Include the full URL."
    )
    education: list[str] = Field(
        description="Each education entry as a SINGLE string in this format: "
                    "'Degree — Institution, Location | Start Date – End Date | GPA/CGPA: X.X (if mentioned) | Relevant Coursework: ... (if mentioned)'. "
                    "Example: 'B.Tech in Computer Science — IIT Bombay, Mumbai | Aug 2019 – May 2023 | CGPA: 8.9 | Coursework: Data Structures, Machine Learning'"
    )
    experience: list[str] = Field(
        description="Each work experience entry as a SINGLE string in this format: "
                    "'Job Title — Company Name, Location | Start Date – End Date\\n• bullet1\\n• bullet2\\n• bullet3'. "
                    "Preserve ALL bullet points/responsibilities exactly as written. Include dates, role title, and company name. "
                    "Example: 'Software Engineer Intern — Google, Bangalore | May 2022 – Jul 2022\\n• Built a microservice handling 10K RPM\\n• Reduced latency by 30%'"
    )
    projects: list[str] = Field(
        default_factory=list,
        description="Each project entry as a SINGLE string in this format: "
                    "'Project Name | Tech Stack: tech1, tech2, tech3 | Date (if available)\\n• bullet1\\n• bullet2'. "
                    "Include the project description, technologies used, and dates if mentioned. "
                    "Example: 'E-Commerce Platform | Tech Stack: React, Node.js, MongoDB | Jan 2023 – Mar 2023\\n• Built a full-stack e-commerce app with payment integration\\n• Deployed on AWS EC2 with CI/CD pipeline'"
    )
    skills: list[str] = Field(
        default_factory=list,
        description="ALL skills exactly as listed on the resume — programming languages, frameworks, libraries, tools, databases, cloud platforms, methodologies, soft skills. "
                    "If the resume organizes skills into categories (e.g., 'Languages: Python, Java'), preserve each skill individually: ['Python', 'Java', ...]. "
                    "Do NOT skip any skill. Include every single one."
    )
    achievements: list[str] = Field(
        default_factory=list,
        description="ALL achievements, awards, certifications, publications, honors, hackathon wins, competitive programming ranks, open source contributions, etc. "
                    "Include the issuing organization and date if mentioned. "
                    "Example: 'AWS Certified Cloud Practitioner — Amazon Web Services, Nov 2024'"
    )




class LatexGen(BaseModel):
    resume: str = Field(
        description="Complete, valid LaTeX source code for a resume. "
                    "Must start with \\documentclass and include "
                    "\\begin{document} and \\end{document}. "
                    "No placeholders, no markdown fences, only raw LaTeX."
    )




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
    model = _create_model(state["api_counter"])
    Keyword_model = model.with_structured_output(KeywordSchema)
    prompt = f"""
You are an elite ATS Keyword Strategist and JD Analyst. Your job is to deeply understand
a Job Description — not just scan it — and extract ONLY the keywords that would actually
appear on a qualified candidate's resume and matter for ATS matching.

═══════════════════════════════════════════════════════════════════
STEP 1: UNDERSTAND THE JD HOLISTICALLY
═══════════════════════════════════════════════════════════════════
Before extracting anything, analyze:
- What is the EXACT role title?
- What industry/domain is this company in?
- What seniority level is this role (Intern/Junior/Mid/Senior/Lead/Staff)?
- What are the 3-5 CORE things this person will do day-to-day?
- What MUST a candidate absolutely know vs. what is nice-to-have?

═══════════════════════════════════════════════════════════════════
STEP 2: EXTRACT MEANINGFUL KEYWORDS (NOT NOISE)
═══════════════════════════════════════════════════════════════════

EXTRACT these (only if they appear or are strongly implied by the JD):

1. **Technical Skills & Tools** — Programming languages, frameworks, libraries, tools,
   platforms, databases, cloud services that are EXPLICITLY mentioned or DIRECTLY implied.
   → e.g., "Python", "React", "Docker", "PostgreSQL", "AWS", "Figma"

2. **Domain-Specific Concepts** — Technical concepts, methodologies, and specialized
   knowledge areas the role requires.
   → e.g., "Machine Learning", "CI/CD", "REST API", "SEO", "A/B Testing", "Data Pipeline"

3. **Role-Relevant Action Competencies** — What the person actually DOES, phrased as
   resume-worthy skills (not vague company culture phrases).
   → e.g., "Content Strategy", "Cross-functional Collaboration", "Technical Writing",
          "System Design", "Performance Optimization", "Stakeholder Management"

4. **Certifications & Standards** — If the JD mentions or implies specific certifications,
   compliance standards, or industry benchmarks.
   → e.g., "AWS Certified", "SOC 2", "GDPR", "PMP"

5. **Soft Skills** — ONLY if the JD explicitly asks for them AND they'd naturally appear
   on a resume's skills section or bullet points.
   → e.g., "Team Leadership", "Mentoring", "Problem-Solving", "Communication"

DO NOT EXTRACT these:
- Company perks, benefits, salary information ("competitive salary", "remote role")
- Vague culture phrases ("supportive environment", "values innovation", "dream big")
- Job posting boilerplate ("we offer", "what we provide", "equal opportunity")
- Candidate traits that aren't skills ("passionate", "self-starter", "thrive in")
- Overly generic filler ("fast-paced", "cutting-edge", "world-class", "delightful")

═══════════════════════════════════════════════════════════════════
STEP 3: INCLUDE STANDARD INDUSTRY COMPANIONS
═══════════════════════════════════════════════════════════════════
If the JD mentions a technology, include its common companions that a qualified
candidate would realistically know:
- "React" → also include "JavaScript", "HTML", "CSS", "TypeScript" (if not already listed)
- "Django" → also include "Python" (if not already listed)
- "Kubernetes" → also include "Docker", "containerization" (if not already listed)
- "Machine Learning" → also include "Deep Learning", "TensorFlow"/"PyTorch" (if relevant to context)

But ONLY add companions that genuinely fit the JD's context. Don't add "Kubernetes"
to a Content Writer JD just because they mentioned "platform".

═══════════════════════════════════════════════════════════════════
STEP 4: ABBREVIATIONS & ALTERNATE FORMS
═══════════════════════════════════════════════════════════════════
Include both the full term AND common abbreviation where applicable:
- "Amazon Web Services" AND "AWS"
- "Continuous Integration/Continuous Deployment" AND "CI/CD"
- "Search Engine Optimization" AND "SEO"
- "User Experience" AND "UX"
- "Application Programming Interface" AND "API"

═══════════════════════════════════════════════════════════════════
JOB DESCRIPTION TO ANALYZE:
═══════════════════════════════════════════════════════════════════
{jd}

Now fill in ALL fields: role_title, industry_domain, seniority_level,
core_responsibilities, hard_skills, soft_skills, domain_keywords, and the
final combined keywords list.

The final `keywords` list should be the UNION of hard_skills + soft_skills + domain_keywords,
deduplicated. Quality over quantity — every keyword should be something a recruiter or ATS
would actually search for when screening resumes for this role.
"""
    response = Keyword_model.invoke(prompt)
    print("=" * 50)
    print(f"Role: {response.role_title} | Domain: {response.industry_domain} | Level: {response.seniority_level}")
    print(f"Core Responsibilities: {response.core_responsibilities}")
    print(f"Hard Skills: {response.hard_skills}")
    print(f"Soft Skills: {response.soft_skills}")
    print(f"Domain Keywords: {response.domain_keywords}")
    print(f"Final Keywords ({len(response.keywords)}): {response.keywords}")
    print("=" * 50)
    return {"keywords": response.keywords}


# ── Node 3: Parse resume details from extracted text ─────────────────
def parse_resume_details(state: AgentState):
    resume_text = state["resume_text"]
    model = _create_model(state["api_counter"])
    Resume_parser_model = model.with_structured_output(ParsedResume)
    prompt = f"""
You are a world-class Resume Parser with 100% recall. Your job is to extract EVERY piece
of information from the raw resume text below — missing even a single detail is a failure.

═══════════════════════════════════════════════════════════════════
RAW RESUME TEXT (extracted from PDF — formatting may be messy):
═══════════════════════════════════════════════════════════════════
\"\"\"
{resume_text}
\"\"\"

═══════════════════════════════════════════════════════════════════
PARSING INSTRUCTIONS — FOLLOW EXACTLY:
═══════════════════════════════════════════════════════════════════

1. **PERSONAL DETAILS**:
   - `name`: The candidate's full name, exactly as written (usually the largest/first text).
   - `email`: Email address (look for patterns like xxx@xxx.xxx).
   - `phone`: Phone number with country code if present (look for +XX, digits, dashes, spaces).
   - `location`: City, State/Province, Country as listed.
   - `links`: Extract ALL URLs — LinkedIn, GitHub, portfolio, personal website, Medium, LeetCode,
     Behance, Dribbble, Kaggle, etc. Include the full URL. Some resumes use icons or labels
     (e.g., "LinkedIn: /in/johndoe") — reconstruct the full URL if possible.

2. **EDUCATION** (CRITICAL — include dates):
   - Extract EVERY education entry: degree, major/field of study, institution name, location, dates.
   - Format each as: "Degree — Institution, Location | Start Date – End Date | GPA: X.X | Coursework: ..."
   - Include GPA/CGPA/percentage if mentioned.
   - Include relevant coursework, honors, or distinctions if listed.
   - If only a graduation year is given, use that as the end date.

3. **EXPERIENCE** (CRITICAL — include dates + ALL bullet points):
   - Extract EVERY work/internship entry: job title, company name, location, start date, end date.
   - Preserve ALL responsibility/achievement bullet points EXACTLY as written — do NOT summarize.
   - Format each as: "Job Title — Company, Location | Start – End\n• bullet1\n• bullet2"
   - If the resume says "Present" or "Current", use "Present" as the end date.
   - Include freelance work, part-time roles, teaching assistantships — everything.

4. **PROJECTS** (CRITICAL — include dates + tech stacks):
   - Extract EVERY project: project name, technologies/tools used, dates (if available).
   - Preserve ALL description bullet points EXACTLY as written.
   - Format each as: "Project Name | Tech Stack: tech1, tech2 | Date\n• bullet1\n• bullet2"
   - Look for tech stacks in parentheses, after "|", after "Built with", or in italic text.
   - If no date is given for a project, omit the date portion but still include everything else.

5. **SKILLS** (CRITICAL — extract EVERY individual skill):
   - Extract ALL skills from every section: "Technical Skills", "Skills", "Core Competencies",
     "Technologies", or skills mentioned in bullet points.
   - Include: programming languages, frameworks, libraries, databases, cloud platforms, DevOps tools,
     operating systems, IDEs, design tools, methodologies (Agile, Scrum), soft skills.
   - If skills are grouped ("Languages: Python, Java, C++"), split them into individual entries.
   - Do NOT skip any skill. Completeness is critical for ATS matching.

6. **ACHIEVEMENTS** (extract ALL):
   - Awards, honors, dean's list, scholarships.
   - Certifications (AWS, Google, Microsoft, etc.) with issuing org and date.
   - Publications, patents, research papers.
   - Hackathon wins, competitive programming ranks (CodeForces, LeetCode, etc.).
   - Open source contributions, speaking engagements.
   - Any other notable accomplishments.
   - Include the issuing organization and date if mentioned.

═══════════════════════════════════════════════════════════════════
CRITICAL RULES:
═══════════════════════════════════════════════════════════════════
- Do NOT invent or hallucinate any information. Only extract what is present in the text.
- Do NOT summarize or paraphrase bullet points — preserve them verbatim.
- Do NOT skip entries because they seem minor — include EVERYTHING.
- Dates are ESSENTIAL for education, experience, and projects. Always extract them.
- If the resume has unusual formatting or sections (e.g., "Volunteering", "Research",
  "Leadership"), map them to the closest field (experience, projects, or achievements).
- If a field has no data in the resume, return an empty string or empty list.
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
\usepackage{mathptmx}  % Times New Roman font
\usepackage[T1]{fontenc}

\definecolor{light-grey}{gray}{0.83}
\definecolor{dark-grey}{gray}{0.3}
\definecolor{text-grey}{gray}{.08}

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
\newcommand{\resumeAchievementListStart}{\begin{itemize}[leftmargin=0.15in]}
\newcommand{\resumeAchievementListEnd}{\end{itemize}\vspace{0pt}}

\color{text-grey}

\begin{document}

%----------HEADER----------
\begin{center}
    \textbf{\Huge Harshibar} \\ \vspace{5pt}
    \small \faPhone* \texttt{555.555.5555} \hspace{1pt} $|$
    \hspace{1pt} \faEnvelope \hspace{2pt} \texttt{hello@email.com} \hspace{1pt} $|$
    \hspace{1pt} \faLinkedin \hspace{2pt} \href{https://linkedin.com/in/harshibar}{\texttt{linkedin.com/in/harshibar}} \hspace{1pt} $|$
    \hspace{1pt} \faGithub \hspace{2pt} \href{https://github.com/harshibar}{\texttt{github.com/harshibar}} \hspace{1pt} $|$
    \hspace{1pt} \faMapMarker* \hspace{2pt}\texttt{San Francisco, CA}
    \\ \vspace{-3pt}
\end{center}

%----------SUMMARY----------
\section{SUMMARY}
 \begin{itemize}[leftmargin=0in, label={}]
    \small{\item{
     Results-driven Software Engineer with 2+ years of experience building scalable web applications and data pipelines. Proficient in Python, JavaScript, and cloud technologies. Passionate about delivering high-impact products that improve user experience and operational efficiency.
    }}
 \end{itemize}

%----------EDUCATION----------
\section{EDUCATION}
  \resumeSubHeadingListStart
    \resumeSubheading
      {Wellesley College}{Aug. 2014 -- May 2018}
      {Bachelor of Arts in Computer Science}{Wellesley, MA}
  \resumeSubHeadingListEnd

%----------EXPERIENCE----------
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

%----------PROJECTS----------
\section{PROJECTS}
    \resumeSubHeadingListStart
      \resumeProjectHeading
          {\textbf{CommonIntern} $|$ \emph{Python, BeautifulSoup, Selenium}}{Sept. 2019 -- May 2020}
          \resumeItemListStart
            \resumeItem{Built a Python script to automatically apply to jobs using BeautifulSoup and Selenium}
          \resumeItemListEnd
    \resumeSubHeadingListEnd

%----------SKILLS----------
\section{SKILLS}
 \begin{itemize}[leftmargin=0in, label={}]
    \small{\item{
     \textbf{Languages} {: Python, JavaScript, HTML/CSS, SQL}\vspace{2pt} \\
     \textbf{Tools}     {: Git, Docker, Jira, Figma}
    }}
 \end{itemize}

%----------ACHIEVEMENTS & CERTIFICATIONS----------
\section{ACHIEVEMENTS \& CERTIFICATIONS}
  \resumeAchievementListStart
    \resumeItem{\textbf{Dean's List} -- Wellesley College, 2014--2018}
    \resumeItem{\textbf{Google Women Techmakers Scholar} -- Google, 2017}
  \resumeAchievementListEnd

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
- FONT: Use Times New Roman via \\usepackage{{mathptmx}}. Do NOT use tgheros, FiraMono, or any sans-serif font.

═══════════════════════════════════════════════════════════════════
MODE & TAILORING INSTRUCTIONS:
═══════════════════════════════════════════════════════════════════
{tailoring_note}

═══════════════════════════════════════════════════════════════════
SECTION ORDER (MANDATORY — follow this EXACT sequence):
═══════════════════════════════════════════════════════════════════
1. HEADER (Name + Personal Details)
2. SUMMARY
3. EDUCATION
4. EXPERIENCE (ONLY if the candidate has work experience — see rule below)
5. PROJECTS
6. SKILLS
7. ACHIEVEMENTS & CERTIFICATIONS

IMPORTANT: If the candidate has NO work experience (no jobs, internships, or professional roles),
do NOT include the EXPERIENCE section at all. Skip it entirely and go directly from EDUCATION to PROJECTS.
Do NOT create a blank or placeholder Experience section.

Do NOT rearrange sections. Follow the above order strictly.

═══════════════════════════════════════════════════════════════════
ATS OPTIMIZATION RULES (CRITICAL — FOLLOW ALL):
═══════════════════════════════════════════════════════════════════

1. HEADER — CLEAN & COMPLETE:
   - Name prominently displayed using \\textbf{{\\Huge ...}}
   - Phone, Email, Location on one line with FontAwesome icons.
   - LinkedIn, GitHub, Portfolio links (if available) on the same or next line.
   - Use \\faPhone*, \\faEnvelope, \\faMapMarker*, \\faLinkedin, \\faGithub icons.
   - Use \\texttt{{}} for contact details and \\href{{}}{{}} for clickable links.

2. SUMMARY SECTION — TAILORED PROFESSIONAL SUMMARY:
   - Write a concise 2-3 sentence professional summary (40-60 words).
   - The summary MUST be specifically tailored to the target JD and role.
   - Mention the candidate's years of experience, core domain, and top 3-4 skills relevant to the JD.
   - Weave in 3-5 HIGH-PRIORITY JD keywords naturally into the summary.
   - Use the format: "[Role-aligned title] with [X]+ years of experience in [domain]. 
     Proficient in [JD-relevant tech/skills]. Proven track record of [key achievement relevant to JD]."
   - ABSOLUTE BLACKLIST for summary (NEVER use these words/phrases):
     "passionate", "self-motivated", "team player", "dynamic", "go-getter", "results-oriented",
     "detail-oriented", "hard-working", "dedicated", "innovative thinker", "synergy",
     "think outside the box", "proactive", "highly motivated", "fast learner", "people person"
   - Instead of buzzwords, use CONCRETE facts: technologies, metrics, and achievements.
   - Do NOT bold anything inside the summary — keep it as clean, flowing text.
   - Example: "Backend Engineer with 3+ years of experience building scalable 
     microservices and RESTful APIs. Proficient in Python, FastAPI, PostgreSQL, and AWS. 
     Proven track record of reducing system latency by 40\\% and processing 2M+ daily transactions."

3. EDUCATION SECTION:
   - Include degree, university, dates, GPA (if > 3.5 / 8.0), relevant coursework.
   - Add relevant coursework ONLY if it matches JD requirements.
   - Bold the institution name and degree — dates in normal weight on the right.

4. EXPERIENCE SECTION — PROFESSIONAL & QUANTIFIED (SKIP IF NO EXPERIENCE):
   - *** CRITICAL: If the candidate has NO work experience, internships, or professional roles,
     do NOT include this section at all. Omit the \\section{{EXPERIENCE}} entirely. ***
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
   - Bold ONLY the company name and job title (in \\resumeSubheading). Do NOT bold random words inside bullets.
   - Bold a key metric or achievement phrase ONLY if it is the most impressive result (max 1 bold per bullet).
   - ANTI-REPETITION: Each bullet MUST start with a DIFFERENT action verb. NEVER repeat the same
     verb across bullets within the same experience entry or across different entries.
     BAD: "Developed X... Developed Y... Developed Z..." (same verb repeated)
     GOOD: "Engineered X... Optimized Y... Deployed Z..." (varied verbs)
   - NEVER use buzzwords in bullets: "dynamic", "synergy", "leveraged", "utilized",
     "facilitated", "passionate", "dedicated". Use precise technical language instead.

5. PROJECTS SECTION — MAXIMUM 3 PROJECTS:
   - CRITICAL RULE: Include AT MOST 3 projects. If the candidate has more than 3, select the 3 most relevant.
   - PROJECT SELECTION LOGIC:
     a) First, check how many of the candidate's projects are RELEVANT to the JD (matching tech stack, domain, or role).
     b) If ALL projects are relevant: pick the top 3 most impressive ones.
     c) If SOME projects are relevant: include ALL relevant ones (up to 3) and fill remaining slots
        with the candidate's best non-relevant projects.
     d) If NO projects are relevant: pick 1-2 candidate projects that are closest in tech/concept,
        and REWRITE their bullet points to emphasize any JD-relevant skills used.
        Then create 1 additional project that uses the JD's tech stack — but ONLY in scratch mode.
        In update mode, use only the candidate's real projects.
   - Format: \\resumeProjectHeading{{\\textbf{{Project Name}} $|$ \\emph{{Tech1, Tech2, Tech3}}}}{{Date}}
   - CRITICAL: List only 3-5 KEY technologies on the heading line — pick the most JD-relevant ones.
     The heading MUST fit on one line. Additional techs can go in the bullets.
   - Each project gets 2-3 strong bullets with quantified results.
   - Bold ONLY the project name. Tech stack in \\emph (italic). Do NOT bold tech names in bullets.

6. SKILLS SECTION — ATS KEYWORD GOLDMINE (CRITICAL):
   - Organize into clear categories using \\textbf for ONLY the category names:
     \\textbf{{Languages}}: Python, Java, JavaScript, SQL
     \\textbf{{Frameworks}}: React, Django, FastAPI, Spring Boot
     \\textbf{{Tools \\& Platforms}}: Docker, Kubernetes, Git, Jenkins
     \\textbf{{Databases}}: PostgreSQL, MongoDB, Redis
     \\textbf{{Cloud}}: AWS, GCP, Azure
   - INCLUDE BOTH:
     a) ALL skills the candidate already has (from their resume).
     b) JD-relevant skills/keywords that the candidate is likely to know based on their experience.
   - DO NOT include buzzwords or filler words that HURT ATS scores. COMPLETE BLACKLIST:
     * Vague traits: "Hard-working", "Team player", "Fast learner", "Detail-oriented", "Self-motivated",
       "Dynamic", "Passionate", "Dedicated", "Innovative", "Proactive", "Go-getter"
     * Generic soft phrases: "Problem-solving", "Communication skills", "Time management",
       "Leadership skills", "Interpersonal skills", "Critical thinking", "Work ethic",
       "Multitasking", "Organizational skills", "Analytical skills"
     * Non-technical filler: "Microsoft Office", "MS Word", "MS Excel", "Google Docs",
       "Google Sheets", "PowerPoint" (unless JD EXPLICITLY requires these)
     * Subjective claims: "Expert in", "Proficient communicator", "Strong team player"
   - The skills section is ONLY for hard skills: tools, technologies, frameworks, languages,
     databases, cloud platforms, methodologies (Agile, Scrum, CI/CD), and technical competencies.
   - ONLY include concrete, searchable, technical skills that ATS scanners look for.
   - Do NOT repeat the same skill in multiple categories.
   - Include 15-25 skills minimum, organized into 3-5 categories.
   - The skills in each category should be listed as comma-separated plain text (not bold).

7. ACHIEVEMENTS & CERTIFICATIONS SECTION:
   - Use \\resumeAchievementListStart and \\resumeAchievementListEnd (NOT \\resumeSubHeadingListStart).
   - This is a DEDICATED command that automatically gives bullet points with proper spacing.
   - DO NOT use \\resumeSubHeadingListStart for achievements. DO NOT use \\resumeItemListStart.
   - Just use \\resumeAchievementListStart, then \\resumeItem entries, then \\resumeAchievementListEnd.
   - EXACT structure to copy (no nesting needed):
     \\resumeAchievementListStart
       \\resumeItem{{\\textbf{{Certificate/Award Name}} -- Issuing Organization, Date}}
       \\resumeItem{{\\textbf{{Award Name}} -- Organization, Date}}
     \\resumeAchievementListEnd
   - Bold ONLY the certificate/award name. Issuing org and date in normal weight.
   - Keep each entry to a SINGLE bullet point. No sub-bullets.
   - The spacing and bullet style will automatically match the experience and projects sections.

═══════════════════════════════════════════════════════════════════
BOLD USAGE RULES (STRICT — DO NOT OVER-BOLD):
═══════════════════════════════════════════════════════════════════
Bold ONLY these elements:
- Candidate's name (\\textbf{{\\Huge ...}})
- Section headings (handled by \\section{{}} command automatically)
- Company names and job titles (in \\resumeSubheading — already bold by template)
- Project names (\\textbf{{Project Name}} in project heading)
- Skills category labels (\\textbf{{Languages}}, \\textbf{{Tools}}, etc.)
- Certificate/award names in achievements
- At most ONE key metric or result per experience bullet (e.g., \\textbf{{saving \\$1M annually}})

Do NOT bold:
- Entire bullet points or sentences
- Technology names inside bullet text
- Dates, locations, or contact details
- Summary text
- Random words for emphasis — if everything is bold, nothing stands out

═══════════════════════════════════════════════════════════════════
SPACING & FORMATTING RULES (EVEN, CLEAN, PROFESSIONAL):
═══════════════════════════════════════════════════════════════════
- The resume MUST be EXACTLY 1 page. Use every inch of space.
- EVEN SPACING: The vertical space between sections MUST be consistent throughout.
  Use the template's built-in spacing commands. Do NOT add extra \\vspace manually.
- Within each section, spacing between entries must also be uniform.
- Do NOT add extra blank lines or \\vspace between items unless the template does it.
- If the content is too sparse, add more bullet points to experience entries.
- If the content overflows, reduce bullets per entry (minimum 2) or trim less relevant entries.
- Use consistent date formatting throughout: "Mon. YYYY -- Mon. YYYY" or "Mon YYYY -- Present".
- Tight margins (already set in template). Do NOT modify margin settings.

═══════════════════════════════════════════════════════════════════
BUZZWORD ELIMINATION (CRITICAL FOR ATS):
═══════════════════════════════════════════════════════════════════
Buzzwords are RESUME POISON. They waste space, add no value, and can REDUCE your ATS score.
Scan the ENTIRE resume and ensure NONE of these appear ANYWHERE (summary, bullets, skills, projects):

FULL BLACKLIST (never use any of these words/phrases):
- dynamic, passionate, dedicated, innovative, proactive, motivated, driven
- self-starter, go-getter, team player, people person, hard-working
- detail-oriented, results-oriented, goal-oriented, solution-oriented
- think outside the box, synergy, leverage, utilize, facilitate
- fast learner, quick learner, eager to learn, willing to learn
- excellent communication skills, strong interpersonal skills
- problem-solving skills, analytical skills, critical thinking
- multitasking, organizational skills, time management
- highly skilled, extensive experience, proven ability
- responsible for, duties included, tasked with

INSTEAD OF BUZZWORDS, always use:
- Specific technologies and tools ("Python", "Docker", "PostgreSQL")
- Quantified achievements ("reduced latency by 35%", "processed 1M+ records")
- Concrete action verbs ("Engineered", "Deployed", "Architected")
- Measurable outcomes ("serving 50K+ users", "\\$200K cost savings")

═══════════════════════════════════════════════════════════════════
WORD REPETITION AVOIDANCE (CRITICAL FOR ATS):
═══════════════════════════════════════════════════════════════════
Repetition makes resumes look lazy and can LOWER ATS scores. Follow these rules strictly:

1. ACTION VERB DIVERSITY: Never use the same action verb more than TWICE across the entire resume.
   - If you've used "Developed" in one bullet, use "Engineered", "Built", "Designed", "Implemented",
     "Created", "Constructed", "Architected" in other bullets.
   - Maintain a mental checklist of verbs already used.

2. KEYWORD VARIETY: Use synonyms and alternate forms to avoid stuffing:
   - Don't write "API" 5 times — alternate with "RESTful services", "endpoints", "web services"
   - Don't write "built" in every bullet — rotate through action verbs
   - Skills section keywords do NOT count as repetition (they are meant to be listed there).

3. SENTENCE STRUCTURE VARIETY: Vary bullet point structures:
   - Mix patterns: "[Verb] [what] using [tech], [result]" and "[Verb] [result] by [method]"
   - Avoid starting every bullet the same way.

4. DO NOT repeat the same metric pattern:
   - BAD: "...by 30%", "...by 40%", "...by 25%", "...by 50%" (same pattern every bullet)
   - GOOD: "...by 30%", "...serving 10K+ users", "...saving \\$50K annually", "...processing 2M records/day"

═══════════════════════════════════════════════════════════════════
ATS SCORE MAXIMIZATION CHECKLIST:
═══════════════════════════════════════════════════════════════════
Before finalizing, mentally verify ALL of the following:
[ ] Summary contains 3-5 JD keywords naturally woven in
[ ] Every experience bullet has a quantified metric
[ ] Skills section contains ALL JD-mentioned technologies
[ ] ZERO buzzwords anywhere in the resume (check against the full blacklist above)
[ ] No action verb is used more than twice across the entire resume
[ ] No word or phrase is unnecessarily repeated
[ ] Metric patterns are varied (not all "by X%")
[ ] Projects showcase JD-relevant tech stacks
[ ] Section headings are standard ATS-readable names (SUMMARY, EDUCATION, EXPERIENCE, PROJECTS, SKILLS, ACHIEVEMENTS & CERTIFICATIONS)
[ ] EXPERIENCE section is OMITTED if the candidate has no work experience
[ ] No tables, columns, or graphics that could confuse ATS parsers (except the template's tabular* for alignment)
[ ] Font is Times New Roman (mathptmx package)
[ ] Bold usage is disciplined — only headings, names, and key metrics
[ ] All keywords from the provided keyword list appear at least once in the resume

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


def _generate_and_compile(prompt, output_dir, Latex_model):
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
    model = _create_model(state["api_counter"])
    Latex_model = model.with_structured_output(LatexGen)
    pdf_path_val = state.get("pdf_path", "")
    output_dir = os.path.dirname(pdf_path_val) if pdf_path_val else tempfile.mkdtemp()
    pdf_path = os.path.join(output_dir, "resume.pdf")

    MAX_RETRIES = 3
    prompt = _build_latex_prompt(parsed_details, jd, keywords, mode=mode)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"[resume_gen] Attempt {attempt}/{MAX_RETRIES}")
        success, latex_code, error_msg = _generate_and_compile(prompt, output_dir, Latex_model)

        if success:
            print("=" * 50)
            print(f"PDF generated successfully at {pdf_path}!")
            print("=" * 50)
            return {"resume": latex_code}

        print(f"[resume_gen] Attempt {attempt} FAILED: {error_msg}")

        if attempt < MAX_RETRIES:
            # Ask the LLM to fix the error on the next attempt
            # Retry with the same Latex_model (same API key for this invocation)
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


def calculate_ats_score(resume_pdf_path: str, jd_text: str, counter: int = 1) -> dict:
    """
    Score a generated resume PDF against a job description.
    Returns a dict with overall_score, breakdown, and suggestions.
    """
    model = _create_model(counter)
    ATS_model = model.with_structured_output(ATSScoreResult)

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
def process_jd(jd_text: str, pdf_path: str = None, mode: str = "update", candidate_details: str = "", counter: int = 1) -> dict:
    """
    Run the full agent pipeline.
    Args:
        jd_text:           The job description text.
        pdf_path:          Absolute path to the user's uploaded resume PDF (None for scratch mode).
        mode:              'scratch' or 'update'
        candidate_details: Structured candidate info from the scratch form (only used in scratch mode).
        counter:           Which API key to use (1-10), cycles round-robin.
    Returns:
        dict with keys: keywords, parsed_details, resume (LaTeX code)
    """
    input_state = {
        "jd_text": jd_text,
        "mode": mode,
        "pdf_path": pdf_path or "",
        "api_counter": counter,
    }
    # In scratch mode, inject the form data as pre-parsed details
    if mode == "scratch" and candidate_details:
        input_state["parsed_details"] = candidate_details
    print(input_state)
    result = agent_app.invoke(input_state)
    return result

