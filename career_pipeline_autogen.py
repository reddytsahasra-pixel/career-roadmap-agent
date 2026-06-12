"""
AI Career Coach — Multi-Agent Pipeline (AG2 / AutoGen, new async API)
=======================================================================

Implements the pipeline:

    User (Target Role + Target Package)
        -> Assessment Agent   (asks questions)
        -> Skill Evaluation Agent (Beginner / Intermediate / Advanced)
        -> Roadmap Agent      (personalized roadmap)
        -> Project Agent      (suggests projects)
        -> Interview Agent    (interview prep plan)
        -> Final Report

This version uses the newer AG2 packages (autogen-agentchat + autogen-ext),
which support Python 3.13. Each "agent" is an AssistantAgent backed by a
shared model client pointed at Gemini's OpenAI-compatible endpoint.

SETUP
-----
1. pip install autogen-agentchat "autogen-ext[openai]" python-dotenv
2. Set your Gemini API key (NOT an OpenAI key):
   - create a .env file in this folder containing:
       GEMINI_API_KEY=your-gemini-key
   - or: $env:GEMINI_API_KEY="your-gemini-key"   (PowerShell)
3. python career_pipeline_autogen.py
"""

import os
import asyncio
from datetime import datetime

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# 1. MODEL CLIENT CONFIGURATION (Gemini via OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set.\n"
        "Create a .env file in this folder containing:\n"
        "  GEMINI_API_KEY=your-gemini-key\n"
        "Get a key from https://aistudio.google.com/api-keys\n"
        "Never hardcode the key directly in source code."
    )

# Gemini exposes an OpenAI-compatible API at this base URL, so we can use
# autogen-ext's OpenAI client to talk to it.
model_client = OpenAIChatCompletionClient(
    model="gemini-2.5-flash-lite",   # swap for any Gemini model your key has access to
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": "unknown",
        "structured_output": False,
    },
)


# ---------------------------------------------------------------------------
# 2. HELPER — single-shot "ask this agent" call
# ---------------------------------------------------------------------------

async def ask(agent: AssistantAgent, prompt: str) -> str:
    """
    Sends a single task to an agent and returns its final text reply.
    """
    result = await agent.run(task=prompt)
    # The last message in the result is the agent's reply
    last_message = result.messages[-1]
    return getattr(last_message, "content", str(last_message))


# ---------------------------------------------------------------------------
# 3. AGENT DEFINITIONS
# ---------------------------------------------------------------------------

assessment_agent = AssistantAgent(
    name="AssessmentAgent",
    model_client=model_client,
    system_message=(
        "You are an Assessment Agent for a career-coaching platform. "
        "Given a target job role and target salary package, generate a focused "
        "skill-assessment quiz for the candidate. "
        "Produce 8-10 questions covering: core language fundamentals, data "
        "structures/algorithms, frameworks relevant to the role, and one "
        "practical/scenario-based question. "
        "Mix question types: a few short-answer conceptual questions and a "
        "couple of 'rate your confidence 1-5' style self-assessment questions. "
        "Number the questions clearly. Do not answer them yourself — only "
        "produce the question set."
    ),
)

skill_eval_agent = AssistantAgent(
    name="SkillEvaluationAgent",
    model_client=model_client,
    system_message=(
        "You are a Skill Evaluation Agent. You receive a set of assessment "
        "questions along with the candidate's answers. "
        "Evaluate the technical correctness, depth, and confidence shown, then "
        "classify the candidate into exactly ONE of: Beginner, Intermediate, "
        "or Advanced. "
        "Respond in this exact format:\n"
        "LEVEL: <Beginner|Intermediate|Advanced>\n"
        "STRENGTHS: <comma separated list>\n"
        "GAPS: <comma separated list>\n"
        "SUMMARY: <2-3 sentence explanation of the rating>"
    ),
)

roadmap_agent = AssistantAgent(
    name="RoadmapAgent",
    model_client=model_client,
    system_message=(
        "You are a Roadmap Agent. Given a target role, target salary package, "
        "the candidate's skill level, strengths, and gaps, generate a "
        "personalized learning roadmap. "
        "Structure it as a week-by-week or phase-by-phase plan (4-12 weeks "
        "depending on the gap between current level and target package "
        "expectations). For each phase include: topics to learn, recommended "
        "resources (by name/type, not fake URLs), and a milestone/checkpoint "
        "to validate progress. Tailor difficulty and pace to the candidate's "
        "current level — don't repeat things they're already strong in."
    ),
)

project_agent = AssistantAgent(
    name="ProjectAgent",
    model_client=model_client,
    system_message=(
        "You are a Project Agent. Given a target role, target package, skill "
        "level, and learning roadmap, suggest 3-5 portfolio projects that "
        "would make the candidate's resume stand out for that role and "
        "package level. "
        "For each project provide: a title, a one-line pitch, the key "
        "skills/tech it demonstrates, and a short list of standout features "
        "that go beyond a basic tutorial-level implementation. "
        "Order projects from foundational to most impressive."
    ),
)

interview_agent = AssistantAgent(
    name="InterviewAgent",
    model_client=model_client,
    system_message=(
        "You are an Interview Preparation Agent. Given a target role, target "
        "package, skill level, and the candidate's gaps, produce an interview "
        "preparation plan. Include: "
        "1) Likely DSA/coding topics and difficulty range to expect for this "
        "package level, "
        "2) Core technical topics/questions specific to the target role, "
        "3) System design / practical topics if the package level warrants it, "
        "4) Behavioral/HR round tips, "
        "5) A suggested practice schedule (e.g. weeks before the interview)."
    ),
)

report_agent = AssistantAgent(
    name="ReportAgent",
    model_client=model_client,
    system_message=(
        "You are a Report Agent. You receive the outputs of an assessment, "
        "skill evaluation, learning roadmap, project suggestions, and interview "
        "prep plan. Compile all of this into ONE polished, well-organized "
        "final report in Markdown, with clear headings for each section: "
        "'Candidate Profile', 'Skill Assessment Result', 'Personalized "
        "Roadmap', 'Recommended Projects', and 'Interview Preparation Plan'. "
        "Keep the original content's substance but improve formatting and "
        "remove redundancy."
    ),
)


# ---------------------------------------------------------------------------
# 4. PIPELINE ORCHESTRATION
# ---------------------------------------------------------------------------

async def run_pipeline(target_role: str, target_package: str):
    print(f"\n=== Career Coaching Pipeline ===")
    print(f"Target Role:    {target_role}")
    print(f"Target Package: {target_package}\n")

    # ---- Step 1: Assessment Agent generates questions ----
    print(">> Generating assessment questions...")
    questions = await ask(
        assessment_agent,
        f"Target Role: {target_role}\nTarget Package: {target_package}\n"
        f"Generate the assessment quiz now.",
    )
    print("\n--- Assessment Questions ---")
    print(questions)

    # ---- Step 2: Collect candidate answers ----
    print("\nPlease answer the questions above.")
    print("Type/paste your answers, then enter a blank line to finish:")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            break
        lines.append(line)
    answers = "\n".join(lines) if lines else "(no answers provided)"

    # ---- Step 3: Skill Evaluation Agent ----
    print("\n>> Evaluating skill level...")
    evaluation = await ask(
        skill_eval_agent,
        f"Questions:\n{questions}\n\nCandidate's Answers:\n{answers}\n\n"
        f"Evaluate now.",
    )
    print("\n--- Skill Evaluation ---")
    print(evaluation)

    # ---- Step 4: Roadmap Agent ----
    print("\n>> Generating personalized roadmap...")
    roadmap = await ask(
        roadmap_agent,
        f"Target Role: {target_role}\nTarget Package: {target_package}\n"
        f"Skill Evaluation:\n{evaluation}\n\nGenerate the roadmap now.",
    )
    print("\n--- Roadmap ---")
    print(roadmap)

    # ---- Step 5: Project Agent ----
    print("\n>> Suggesting projects...")
    projects = await ask(
        project_agent,
        f"Target Role: {target_role}\nTarget Package: {target_package}\n"
        f"Skill Evaluation:\n{evaluation}\n\nRoadmap:\n{roadmap}\n\n"
        f"Suggest projects now.",
    )
    print("\n--- Project Suggestions ---")
    print(projects)

    # ---- Step 6: Interview Agent ----
    print("\n>> Building interview preparation plan...")
    interview_plan = await ask(
        interview_agent,
        f"Target Role: {target_role}\nTarget Package: {target_package}\n"
        f"Skill Evaluation:\n{evaluation}\n\nGenerate the interview prep plan now.",
    )
    print("\n--- Interview Preparation Plan ---")
    print(interview_plan)

    # ---- Step 7: Final Report ----
    print("\n>> Compiling final report...")
    final_report = await ask(
        report_agent,
        "Compile the final report from these sections:\n\n"
        f"## Target Role\n{target_role}\n\n"
        f"## Target Package\n{target_package}\n\n"
        f"## Assessment Questions\n{questions}\n\n"
        f"## Candidate Answers\n{answers}\n\n"
        f"## Skill Evaluation\n{evaluation}\n\n"
        f"## Roadmap\n{roadmap}\n\n"
        f"## Projects\n{projects}\n\n"
        f"## Interview Plan\n{interview_plan}\n",
    )

    # ---- Save report ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"final_report_{timestamp}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(final_report)

    print(f"\n=== Final report saved to {out_path} ===")

    await model_client.close()
    return final_report


# ---------------------------------------------------------------------------
# 5. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    target_role = input("Enter target role [Python Developer]: ").strip() or "Python Developer"
    target_package = input("Enter target package [12 LPA]: ").strip() or "12 LPA"

    asyncio.run(run_pipeline(target_role, target_package))