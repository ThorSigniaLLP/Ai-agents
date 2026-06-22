"""
Multi-Model Research Pipeline — Fault-Tolerant Architecture
============================================================
ARCHITECTURE (Graceful Degradation):

  Each researcher runs INDEPENDENTLY as a direct LLM call (not CrewAI agent).
  If any one (or more) researchers fail → they are skipped gracefully.
  Research continues with all successfully returned reports.
  The editor only receives reports that actually succeeded.

  RESEARCHERS (run independently, collected into a list):
    1. Gemini 2.5 Flash          — LIVE Google Search (real-time web data)
    2. Groq Llama 3.3 70B        — fast, broad general knowledge
    3. Groq Llama 4 Scout 17B    — Meta's latest multimodal model
    4. Groq Qwen3 32B            — strong reasoning / tech coverage
    5. Cerebras GPT-OSS 120B     — OSS-trained knowledge
    6. Cerebras ZAI-GLM 4.7      — international / Asia-Pacific coverage
    7. OpenRouter NVIDIA 120B    — NVIDIA-trained enterprise knowledge
    8. OpenRouter Gemma 4 31B    — Google's knowledge base

  EDITOR (CrewAI Agent):
    OpenRouter Gemma 4 31B — merges all successful reports into clean JSON

  FLOW:
    ┌────────────────────────────────────────────────────────┐
    │  Run 8 researchers independently (try/except each)     │
    │  Collect all that succeed (minimum 1 required)         │
    └───────────────────────┬────────────────────────────────┘
                            │
                            ▼
    ┌────────────────────────────────────────────────────────┐
    │  Editor: merge → deduplicate → clean JSON              │
    └────────────────────────────────────────────────────────┘

  NO TAVILY · NO PITCH WRITING · NO SINGLE POINT OF FAILURE
"""
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

from crewai import Agent, Task, Crew, Process, LLM
import google.genai as genai_client
from google.genai import types as genai_types
from litellm import completion as litellm_completion

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _build_card_block(card: dict) -> str:
    field_labels = [
        ("name",      "Contact Name"),
        ("company",   "Company"),
        ("job_title", "Job Title"),
        ("email",     "Email"),
        ("mobile",    "Mobile"),
        ("website",   "Website"),
        ("address",   "Address / Location"),
        ("notes",     "Additional Notes"),
    ]
    lines = []
    for key, label in field_labels:
        val = (card.get(key) or "").strip()
        if val:
            lines.append(f"  - {label}: {val}")
    return "\n".join(lines) if lines else "  (no card fields provided)"


# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH PROMPT (shared by all non-Gemini researchers)
# ─────────────────────────────────────────────────────────────────────────────
def _build_research_prompt(company: str, card_block: str) -> str:
    return f"""
You are a business intelligence researcher. Using your training knowledge,
report everything you know about the company described below.

Business Card Details:
{card_block}

Research and report on {company}:
1.  Company overview — industry, founding year, HQ location
2.  Products & services — key offerings, recent launches
3.  Company size — employees, revenue estimate
4.  Leadership — CEO and key executives
5.  Funding & investors (if applicable)
6.  Target customers and industries served
7.  Top 3-5 direct competitors with brief descriptions
8.  Recent news or milestones (last 1-2 years) with dates
9.  Strategic priorities and growth direction
10. Known challenges, controversies, or market headwinds

Be specific. Include numbers, dates, and names wherever possible.
Mark confidence: [HIGH], [MEDIUM], or [LOW] after each fact.
Only report what you genuinely know — do NOT invent facts.
"""


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL RESEARCHER FUNCTIONS (each wrapped in try/except)
# ─────────────────────────────────────────────────────────────────────────────

def _researcher_gemini_search(company: str, card_block: str, key: str) -> Optional[str]:
    """Gemini 2.5 Flash with LIVE Google Search grounding."""
    try:
        client = genai_client.Client(api_key=key)
        prompt = f"""
You are a business intelligence researcher with live web search access.
Use Google Search to find real, current information about the company below.

Business Card Details:
{card_block}

Find and report (use web search actively):
1.  Company overview — what they do, industry, founding year, HQ
2.  Products & services — main offerings, key features, recent launches
3.  Company size — employee count, revenue (latest available)
4.  Leadership — CEO and key executives (current)
5.  Funding & investors — rounds, total raised (if startup/scaleup)
6.  Target customers — industries or personas served
7.  Top 3-5 direct competitors
8.  Recent news & milestones — last 12 months (include dates)
9.  Strategic priorities — growth plans, product roadmap, M&A
10. Known challenges or controversies

Mark each fact: [LIVE-SEARCH] if confirmed from web, [KNOWLEDGE] if from training data.
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
            )
        )
        return response.text
    except Exception as e:
        logger.warning(f"[Researcher] Gemini Search FAILED: {e}")
        return None


def _researcher_litellm(label: str, model: str, api_key: str,
                        company: str, card_block: str, extra_env: dict = None) -> Optional[str]:
    """Generic LiteLLM researcher. Returns report string or None on failure."""
    try:
        prompt = _build_research_prompt(company, card_block)
        # Some providers need env vars set
        if extra_env:
            for k, v in extra_env.items():
                os.environ[k] = v

        response = litellm_completion(
            model=model,
            api_key=api_key,
            messages=[{"role": "user", "content": prompt}],
            timeout=60
        )
        text = response.choices[0].message.content.strip()
        logger.info(f"[Researcher] {label} OK — {len(text)} chars")
        return text
    except Exception as e:
        logger.warning(f"[Researcher] {label} FAILED: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_research_test_crew(card_info: dict) -> dict:
    """
    Fault-tolerant multi-model research pipeline.
    Accepts a business card dict, returns a clean structured research profile.

    Graceful degradation: if any model fails, research continues with the rest.
    Minimum 1 successful researcher required; editor always runs if ≥1 succeed.
    """
    company    = card_info.get("company", "").strip()
    card_block = _build_card_block(card_info)

    gemini_key     = os.getenv("GEMINI_API_KEY")
    groq_key       = os.getenv("GROQ_API_KEY")
    cerebras_key   = os.getenv("cerebras_api_key") or os.getenv("CEREBRAS_API_KEY")
    openrouter_key = os.getenv("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY")

    # ── Define all researchers ─────────────────────────────────────────────────
    # Each entry: (label, callable)
    researcher_jobs = [
        ("Gemini 2.5 Flash [LIVE SEARCH]",
         lambda: _researcher_gemini_search(company, card_block, gemini_key)),

        ("Groq Llama 3.3 70B",
         lambda: _researcher_litellm("Groq Llama 3.3 70B", "groq/llama-3.3-70b-versatile",
                                     groq_key, company, card_block)),

        ("Groq Llama 4 Scout 17B",
         lambda: _researcher_litellm("Groq Llama 4 Scout", "groq/meta-llama/llama-4-scout-17b-16e-instruct",
                                     groq_key, company, card_block)),

        ("Groq Qwen3 32B",
         lambda: _researcher_litellm("Groq Qwen3 32B", "groq/qwen/qwen3-32b",
                                     groq_key, company, card_block)),

        ("Cerebras GPT-OSS 120B",
         lambda: _researcher_litellm("Cerebras GPT-OSS 120B", "cerebras/gpt-oss-120b",
                                     cerebras_key, company, card_block)),

        ("Cerebras ZAI-GLM 4.7",
         lambda: _researcher_litellm("Cerebras ZAI-GLM 4.7", "cerebras/zai-glm-4.7",
                                     cerebras_key, company, card_block)),

        ("OpenRouter NVIDIA Nemotron 120B",
         lambda: _researcher_litellm("OpenRouter NVIDIA 120B",
                                     "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
                                     openrouter_key, company, card_block)),

        ("OpenRouter Gemma 4 31B",
         lambda: _researcher_litellm("OpenRouter Gemma 4 31B",
                                     "openrouter/google/gemma-4-31b-it:free",
                                     openrouter_key, company, card_block)),
    ]

    # ── Run all researchers in parallel ──────────────────────────────────────
    logger.info(f"[ResearchTest] Starting {len(researcher_jobs)} researchers for: {company}")
    successful_reports: Dict[str, str] = {}
    failed_models: list = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_label = {
            executor.submit(fn): label
            for label, fn in researcher_jobs
        }
        for future in as_completed(future_to_label):
            label = future_to_label[future]
            try:
                result = future.result()
                if result:
                    successful_reports[label] = result
                    logger.info(f"[ResearchTest] OK {label} -- {len(result)} chars")
                else:
                    failed_models.append(label)
                    logger.warning(f"[ResearchTest] FAIL {label} -- returned empty/None")
            except Exception as e:
                failed_models.append(label)
                logger.warning(f"[ResearchTest] FAIL {label} -- exception: {e}")

    logger.info(f"[ResearchTest] Completed: {len(successful_reports)} succeeded, {len(failed_models)} failed")

    if not successful_reports:
        raise RuntimeError(
            "All research agents failed. No data to process. "
            f"Failed models: {failed_models}"
        )

    # ── Build combined report for editor ─────────────────────────────────────
    combined_report_parts = []
    for i, (label, report) in enumerate(successful_reports.items(), 1):
        combined_report_parts.append(
            f"=== SOURCE {i}: {label} ===\n{report}\n"
        )
    combined_report = "\n".join(combined_report_parts)

    # ── Editor Agent (CrewAI — receives all successful reports) ──────────────
    editor_llm = LLM(
        model="openrouter/google/gemma-4-31b-it:free",
        temperature=0,
        api_key=openrouter_key
    )

    editor_agent = Agent(
        role="Senior Intelligence Editor",
        goal=f"Merge {len(successful_reports)} research reports about {company} into one clean, deduplicated JSON profile",
        backstory=f"""
You are a meticulous intelligence editor. You have received {len(successful_reports)} independent
research reports about the same company, each from a different AI model with different training data.

Your job:
1. Keep every UNIQUE fact — even if only one source mentions it
2. When the same fact appears in multiple sources → keep the most detailed version, discard duplicates
3. Prioritize [LIVE-SEARCH] facts (from Gemini) over [KNOWLEDGE] facts when they conflict
4. Flag genuine conflicts: ⚠️ CONFLICT: [source A: X] vs [source B: Y]
5. Remove speculation, invented data, or very low-confidence claims
6. Output a single clean, structured JSON profile
""",
        verbose=True,
        allow_delegation=False,
        llm=editor_llm
    )

    editor_task = Task(
        description=f"""
You have received {len(successful_reports)} independent research reports about "{company}".
Data sources that succeeded: {list(successful_reports.keys())}
Data sources that failed (already handled gracefully): {failed_models}

Here are all the research reports:

{combined_report}

Merge them into one clean, structured JSON profile. Follow the expected output format exactly.
""",
        expected_output="""
A single clean JSON object with these keys:
{
  "company_name": "",
  "overview": "",
  "products_services": [""],
  "company_size": "",
  "hq_location": "",
  "founding_year": "",
  "leadership": [{"name": "", "role": ""}],
  "funding": "",
  "target_customers": [""],
  "competitors": [{"name": "", "description": ""}],
  "recent_news": [""],
  "strategic_priorities": [""],
  "pain_points": [""],
  "live_search_facts": [""],
  "conflicts_detected": [""],
  "data_sources_succeeded": [],
  "data_sources_failed": [],
  "confidence_summary": ""
}
""",
        agent=editor_agent
    )

    crew = Crew(
        agents=[editor_agent],
        tasks=[editor_task],
        process=Process.sequential,
        verbose=True,
        memory=False
    )

    result = crew.kickoff()

    return {
        "status": "success",
        "company": company,
        "researchers_succeeded": list(successful_reports.keys()),
        "researchers_failed": failed_models,
        "total_raw_chars_collected": sum(len(r) for r in successful_reports.values()),
        "research_output": str(result)
    }
