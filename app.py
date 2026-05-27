"""
CV Analyzer Web App
====================
A Streamlit-based HR tool that uses Google Gemini 1.5 Flash to evaluate
CVs against a job description and produce structured, scored reports.

Tech Stack:
- Frontend/Backend : Streamlit
- LLM              : Google Gemini 1.5 Flash  (google-genai SDK)
- Validation       : Pydantic
- PDF Parsing      : PyMuPDF (fitz)
- DOCX Parsing     : python-docx
- Data Handling    : Pandas
"""

import io
import time

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Output Schema
# ──────────────────────────────────────────────────────────────────────────────

class CVEvaluation(BaseModel):
    """Strictly typed schema for a single CV evaluation result."""

    candidate_name: str = Field(description="Full name of the candidate extracted from the CV.")
    match_score: int = Field(
        ge=0, le=100,
        description="Integer score from 0 to 100 representing how well the CV matches the JD."
    )
    key_expertise: list[str] = Field(
        description="List of the candidate's skills/experiences that are relevant to the JD."
    )
    missing_skills: list[str] = Field(
        description="List of skills or qualifications required by the JD that are absent from the CV."
    )
    hr_recommendation: str = Field(
        description="One of exactly three values: 'Shortlist', 'Keep on File', or 'Reject'."
    )
    brief_justification: str = Field(
        description="A concise 1–2 sentence justification for the recommendation."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Document Extraction Helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract all text from a PDF file using PyMuPDF.

    Args:
        file_bytes: Raw bytes of the PDF file.

    Returns:
        Concatenated text from all pages, or an empty string on failure.
    """
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            pages_text = []
            for page in doc:
                text = page.get_text("text")
                if text and text.strip():
                    pages_text.append(text)
            return "\n".join(pages_text)
    except Exception as e:
        st.warning(f"⚠️ Could not parse PDF: {e}")
        return ""


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract all paragraph text from a DOCX file using python-docx.

    Args:
        file_bytes: Raw bytes of the DOCX file.

    Returns:
        Concatenated paragraph text, or an empty string on failure.
    """
    try:
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        st.warning(f"⚠️ Could not parse DOCX: {e}")
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# AI Evaluation Engine
# ──────────────────────────────────────────────────────────────────────────────

EVALUATION_PROMPT_TEMPLATE = """
You are an expert, impartial HR analyst. Your task is to evaluate the provided
CV against the Job Description (JD) and return a structured JSON evaluation.

<JOB_DESCRIPTION>
{jd_text}
</JOB_DESCRIPTION>

<CANDIDATE_CV>
{cv_text}
</CANDIDATE_CV>

Evaluation Guidelines:
1. **Be Fair and Practical**: Do not reject a candidate over minor phrasing differences. Use semantic matching (e.g., if the JD asks for "React", and the CV says "React.js" or "frontend developer with modern JS frameworks," consider it a match).
2. **Core vs. Nice-to-Have**: Give significantly more weight to core requirements (e.g., primary programming language, core domain experience) than to nice-to-have/peripheral skills.
3. **Experience Equivalence**: Recognize equivalent experience levels (e.g., "5 years" required can match a highly accomplished candidate with "4 years" or a strong portfolio).

Instructions:
1. Extract the candidate's full name from the CV. If not found, use a reasonable placeholder or the file name.
2. Assign an integer `match_score` from 0–100 based on how well the CV aligns with the JD.
3. List the relevant skills/experience the candidate *does* have (`key_expertise`).
4. List primary JD requirements the candidate is *actually missing* (`missing_skills`).
5. Provide an `hr_recommendation`: ONLY one of "Shortlist", "Keep on File", or "Reject".
   - Shortlist  : score >= 70 (Strong match on core requirements)
   - Keep on File: score 40–69 (Good potential, but missing some key core requirements)
   - Reject     : score < 40 (Significant gap in core requirements)
6. Write a concise `brief_justification` (max 2 sentences).

Return ONLY valid JSON conforming to the schema. Do not include markdown fences.
"""


def evaluate_cv_with_gemini(
    cv_text: str,
    jd_text: str,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
) -> CVEvaluation | None:
    """
    Send the CV + JD to the specified Gemini model and parse the structured response.

    Args:
        cv_text : Extracted plain text from the CV.
        jd_text : The job description pasted by the HR user.
        api_key : Google Gemini API key provided by the user.
        model_name : Gemini model name string to use for generation.

    Returns:
        A validated CVEvaluation Pydantic object, or None on failure.
    """
    try:
        # Initialise the new google-genai client with the user-supplied key
        client = genai.Client(api_key=api_key)

        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            jd_text=jd_text.strip(),
            cv_text=cv_text.strip(),
        )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CVEvaluation,
                temperature=0.1,   # Low temp → deterministic, analytical output
            ),
        )

        # Parse the validated Pydantic object from the response
        evaluation: CVEvaluation = response.parsed
        return evaluation

    except genai.errors.APIError as api_err:
        st.error(f"🔴 Gemini API Error: {api_err}")
        return None
    except Exception as exc:
        st.error(f"🔴 Unexpected error during evaluation: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI & Application Logic
# ──────────────────────────────────────────────────────────────────────────────

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CV Analyzer | AI-Powered HR Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* ── Global font & background ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }

        /* ── Main header ── */
        .hero-title {
            font-size: 2.6rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.25rem;
        }
        .hero-subtitle {
            color: #64748b;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        /* ── Score badge colours ── */
        .badge-shortlist  { background:#16a34a; color:#fff; padding:3px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; }
        .badge-keeponfile { background:#d97706; color:#fff; padding:3px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; }
        .badge-reject     { background:#dc2626; color:#fff; padding:3px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; }

        /* ── Run button ── */
        div.stButton > button {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            font-weight: 600;
            font-size: 1rem;
            padding: 0.6rem 2rem;
            border: none;
            border-radius: 8px;
            transition: opacity 0.2s ease;
        }
        div.stButton > button:hover { opacity: 0.88; }

        /* ── Expander ── */
        details summary { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    api_key = st.text_input(
        "🔑 Google Gemini API Key",
        type="password",
        placeholder="Paste your API key here…",
        help="Your key is used only for this session and never stored.",
    )

    st.markdown("---")

    # Dynamic Model Selector
    model_option = st.selectbox(
        "🤖 AI Model Selection",
        options=[
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash (legacy)",
            "Other / Custom Model"
        ],
        index=0,
        help="Select the Gemini model to use for the evaluation. 'gemini-2.5-flash' or 'gemini-2.0-flash' are recommended."
    )

    if model_option == "Other / Custom Model":
        model_name = st.text_input(
            "Enter Custom Model Name",
            value="gemini-2.5-flash",
            help="Enter the exact model identifier from Google AI Studio."
        )
    else:
        model_name = model_option.split(" ")[0]

    st.markdown("---")
    st.markdown(
        """
        **How it works**
        1. Enter your Gemini API key above.
        2. Paste the job description.
        3. Upload one or more CVs (PDF / DOCX).
        4. Click **Run Analysis**.

        **Rate limit note** — A 2-second delay is added between each CV to
        stay within the Gemini free-tier limit (15 RPM).
        """
    )
    st.markdown("---")
    st.caption("Powered by Google Gemini · google-genai SDK")

# ── Main area header ──────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎯 CV Analyzer Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">AI-powered CV screening — upload CVs, paste a JD, and get ranked results in seconds.</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── Job Description input ──────────────────────────────────────────────────────
st.subheader("📋 Job Description")
job_description = st.text_area(
    label="Paste the full job description here",
    height=220,
    placeholder=(
        "e.g.\n"
        "We are looking for a Senior Python Engineer with 5+ years of experience "
        "in FastAPI, PostgreSQL, and cloud deployment (AWS/GCP). Familiarity with "
        "LLM integration and MLOps pipelines is a strong advantage…"
    ),
    label_visibility="collapsed",
)

st.divider()

# ── File uploader ──────────────────────────────────────────────────────────────
st.subheader("📁 Upload CVs")
uploaded_files = st.file_uploader(
    label="Upload candidate CVs",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    help="Accepts PDF and DOCX files. You may upload multiple files at once.",
    label_visibility="collapsed",
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} file(s) ready for analysis.")

st.divider()

# ── Run Analysis button ────────────────────────────────────────────────────────
run_analysis = st.button("🚀 Run Analysis", use_container_width=False)

# ── Processing & Results ───────────────────────────────────────────────────────
if run_analysis:
    # ── Input validation ──────────────────────────────────────────────────────
    if not api_key:
        st.error("❌ Please enter your Gemini API key in the sidebar before running.")
        st.stop()

    if not job_description.strip():
        st.error("❌ Please paste a Job Description before running.")
        st.stop()

    if not uploaded_files:
        st.error("❌ Please upload at least one CV file.")
        st.stop()

    # ── Batch processing loop ─────────────────────────────────────────────────
    st.subheader("⏳ Processing CVs…")
    progress_bar = st.progress(0, text="Starting analysis…")

    results: list[dict] = []
    total = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        file_name = uploaded_file.name
        progress_text = f"Analysing **{file_name}** ({idx + 1}/{total})…"
        progress_bar.progress((idx) / total, text=progress_text)

        with st.spinner(progress_text):
            # Extract text based on file extension
            file_bytes = uploaded_file.read()
            if file_name.lower().endswith(".pdf"):
                cv_text = extract_text_from_pdf(file_bytes)
            elif file_name.lower().endswith(".docx"):
                cv_text = extract_text_from_docx(file_bytes)
            else:
                st.warning(f"⚠️ Unsupported file type: {file_name}. Skipping.")
                continue

            if not cv_text.strip():
                st.warning(f"⚠️ No text could be extracted from **{file_name}**. Skipping.")
                continue

            # Call the Gemini evaluation engine
            evaluation = evaluate_cv_with_gemini(cv_text, job_description, api_key, model_name=model_name)

            if evaluation is not None:
                results.append({
                    "File": file_name,
                    "Candidate Name": evaluation.candidate_name,
                    "Match Score (%)": evaluation.match_score,
                    "Recommendation": evaluation.hr_recommendation,
                    "Key Expertise": evaluation.key_expertise,
                    "Missing Skills": evaluation.missing_skills,
                    "Justification": evaluation.brief_justification,
                    "Extracted Text": cv_text,
                })

        # ── Rate-limit guard: 2-second pause between requests ─────────────────
        if idx < total - 1:
            time.sleep(2)

    progress_bar.progress(1.0, text="✅ Analysis complete!")

    # ── Results display ───────────────────────────────────────────────────────
    if not results:
        st.warning("No valid results were returned. Please check your API key and uploaded files.")
        st.stop()

    st.divider()
    st.subheader(f"📊 Results — {len(results)} Candidate(s) Evaluated")

    # Build the summary DataFrame (sorted by score, highest first)
    df = pd.DataFrame(results)
    df_display = (
        df[["Candidate Name", "File", "Match Score (%)", "Recommendation"]]
        .sort_values("Match Score (%)", ascending=False)
        .reset_index(drop=True)
    )

    # Style score column with colour gradient
    styled_df = df_display.style.background_gradient(
        subset=["Match Score (%)"],
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
    ).format({"Match Score (%)": "{:.0f}"})

    st.dataframe(styled_df, use_container_width=True, height=min(400, 55 + 35 * len(df_display)))

    st.divider()
    st.subheader("🔍 Candidate Detail Cards")

    # Sort detail cards by score (highest first) to match the table
    sorted_results = sorted(results, key=lambda x: x["Match Score (%)"], reverse=True)

    for candidate in sorted_results:
        name = candidate["Candidate Name"]
        score = candidate["Match Score (%)"]
        rec = candidate["Recommendation"]
        missing = candidate["Missing Skills"]
        key_exp = candidate["Key Expertise"]
        justification = candidate["Justification"]
        file = candidate["File"]

        # Choose badge class based on recommendation
        rec_lower = rec.lower().replace(" ", "")
        badge_class = {
            "shortlist": "badge-shortlist",
            "keeponfile": "badge-keeponfile",
            "reject": "badge-reject",
        }.get(rec_lower, "badge-keeponfile")

        expander_label = (
            f"{name}  |  Score: {score}/100  |  {rec}  ·  ({file})"
        )

        with st.expander(expander_label, expanded=(score >= 70)):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**✅ Key Expertise**")
                if key_exp:
                    for skill in key_exp:
                        st.markdown(f"- {skill}")
                else:
                    st.markdown("_None identified._")

            with col2:
                st.markdown("**❌ Missing Skills**")
                if missing:
                    for skill in missing:
                        st.markdown(f"- {skill}")
                else:
                    st.markdown("_No critical gaps detected._")

            st.markdown("---")
            st.markdown(f"**💬 Justification:** {justification}")
            st.markdown(
                f'<span class="{badge_class}">{rec}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📄 View Extracted CV Text (For Verification)", expanded=False):
                st.text_area(
                    label="Parsed Text",
                    value=candidate.get("Extracted Text", ""),
                    height=200,
                    disabled=True,
                    label_visibility="collapsed"
                )

    # ── CSV Download ──────────────────────────────────────────────────────────
    st.divider()
    csv_data = df_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results as CSV",
        data=csv_data,
        file_name="cv_analysis_results.csv",
        mime="text/csv",
        use_container_width=False,
    )
