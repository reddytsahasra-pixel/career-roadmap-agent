"""
AI Career Coach — Streamlit UI (AG2 / AutoGen, new async API + Gemini)
=========================================================================

Run with:
    pip install streamlit autogen-agentchat "autogen-ext[openai]" python-dotenv
    streamlit run app.py

The app walks the user through:

    Target Role + Target Package
        -> Assessment Agent (questions)
        -> [user answers in the UI]
        -> Skill Evaluation Agent (Beginner/Intermediate/Advanced)
        -> Roadmap Agent
        -> Project Agent
        -> Interview Agent
        -> Final Report (downloadable Markdown)

Uses Gemini via its OpenAI-compatible endpoint, same as career_pipeline_autogen.py.
"""

import os
import asyncio
from datetime import datetime

import streamlit as st
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AI Career Coach", page_icon="🧭", layout="wide")


# ---------------------------------------------------------------------------
# SIDEBAR — LLM CONFIG
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Settings")

api_key_input = st.sidebar.text_input(
    "Gemini API Key",
    value=os.environ.get("GEMINI_API_KEY", ""),
    type="password",
    help="Get one from https://aistudio.google.com/api-keys. "
         "Or set the GEMINI_API_KEY environment variable before launching.",
)
model_name = st.sidebar.text_input("Model", value="gemini-2.5-flash-lite")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Restart Pipeline"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ---------------------------------------------------------------------------
# AGENT SETUP (cached so we don't rebuild on every rerun)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def build_agents(api_key: str, model: str):
    model_client = OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        },
    )

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

    return {
        "model_client": model_client,
        "assessment": assessment_agent,
        "skill_eval": skill_eval_agent,
        "roadmap": roadmap_agent,
        "project": project_agent,
        "interview": interview_agent,
        "report": report_agent,
    }


def ask(agent: AssistantAgent, prompt: str) -> str:
    """Run a single-shot task against an agent (sync wrapper around async run)."""
    async def _run():
        result = await agent.run(task=prompt)
        last_message = result.messages[-1]
        return getattr(last_message, "content", str(last_message))

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------------------------------

if "stage" not in st.session_state:
    st.session_state.stage = "input"
if "data" not in st.session_state:
    st.session_state.data = {}


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------

st.title("🧭 AI Career Coach")
st.caption(
    "Assessment → Skill Evaluation → Roadmap → Projects → Interview Prep → Final Report"
)

stages = ["input", "assessment", "results"]
stage_labels = {"input": "1. Target Setup", "assessment": "2. Skill Assessment", "results": "3. Your Plan"}
current_idx = stages.index(st.session_state.stage)
st.progress((current_idx + 1) / len(stages), text=stage_labels[st.session_state.stage])


# ---------------------------------------------------------------------------
# STAGE 1: INPUT TARGET ROLE / PACKAGE
# ---------------------------------------------------------------------------

if st.session_state.stage == "input":
    st.subheader("Tell us your goal")

    col1, col2 = st.columns(2)
    with col1:
        role = st.text_input("Target Role", value="Python Developer")
    with col2:
        package = st.text_input("Target Package", value="12 LPA")

    if st.button("🚀 Start Assessment", type="primary"):
        if not api_key_input:
            st.error("Please enter your Gemini API key in the sidebar.")
        else:
            agents = build_agents(api_key_input, model_name)
            with st.spinner("Assessment Agent is preparing your quiz..."):
                questions = ask(
                    agents["assessment"],
                    f"Target Role: {role}\nTarget Package: {package}\n"
                    f"Generate the assessment quiz now.",
                )
            st.session_state.data["role"] = role
            st.session_state.data["package"] = package
            st.session_state.data["questions"] = questions
            st.session_state.stage = "assessment"
            st.rerun()


# ---------------------------------------------------------------------------
# STAGE 2: SHOW QUESTIONS, COLLECT ANSWERS
# ---------------------------------------------------------------------------

elif st.session_state.stage == "assessment":
    data = st.session_state.data
    st.subheader(f"Skill Assessment — {data['role']} ({data['package']})")

    with st.expander("📋 Assessment Questions", expanded=True):
        st.markdown(data["questions"])

    answers = st.text_area(
        "✍️ Your Answers (answer each question, numbered the same way)",
        height=300,
        placeholder="1. ...\n2. ...\n3. ...",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        back = st.button("⬅️ Back")
    with col2:
        submit = st.button("✅ Submit Answers", type="primary")

    if back:
        st.session_state.stage = "input"
        st.rerun()

    if submit:
        if not answers.strip():
            st.warning("Please answer at least some of the questions before submitting.")
        else:
            agents = build_agents(api_key_input, model_name)
            data["answers"] = answers

            with st.spinner("Skill Evaluation Agent is reviewing your answers..."):
                data["evaluation"] = ask(
                    agents["skill_eval"],
                    f"Questions:\n{data['questions']}\n\n"
                    f"Candidate's Answers:\n{answers}\n\nEvaluate now.",
                )

            with st.spinner("Roadmap Agent is building your personalized roadmap..."):
                data["roadmap"] = ask(
                    agents["roadmap"],
                    f"Target Role: {data['role']}\nTarget Package: {data['package']}\n"
                    f"Skill Evaluation:\n{data['evaluation']}\n\nGenerate the roadmap now.",
                )

            with st.spinner("Project Agent is suggesting portfolio projects..."):
                data["projects"] = ask(
                    agents["project"],
                    f"Target Role: {data['role']}\nTarget Package: {data['package']}\n"
                    f"Skill Evaluation:\n{data['evaluation']}\n\n"
                    f"Roadmap:\n{data['roadmap']}\n\nSuggest projects now.",
                )

            with st.spinner("Interview Agent is preparing your interview plan..."):
                data["interview"] = ask(
                    agents["interview"],
                    f"Target Role: {data['role']}\nTarget Package: {data['package']}\n"
                    f"Skill Evaluation:\n{data['evaluation']}\n\n"
                    f"Generate the interview prep plan now.",
                )

            with st.spinner("Report Agent is compiling your final report..."):
                data["final_report"] = ask(
                    agents["report"],
                    "Compile the final report from these sections:\n\n"
                    f"## Target Role\n{data['role']}\n\n"
                    f"## Target Package\n{data['package']}\n\n"
                    f"## Assessment Questions\n{data['questions']}\n\n"
                    f"## Candidate Answers\n{answers}\n\n"
                    f"## Skill Evaluation\n{data['evaluation']}\n\n"
                    f"## Roadmap\n{data['roadmap']}\n\n"
                    f"## Projects\n{data['projects']}\n\n"
                    f"## Interview Plan\n{data['interview']}\n",
                )

            st.session_state.stage = "results"
            st.rerun()


# ---------------------------------------------------------------------------
# STAGE 3: RESULTS
# ---------------------------------------------------------------------------

elif st.session_state.stage == "results":
    data = st.session_state.data
    st.subheader(f"Your Personalized Plan — {data['role']} ({data['package']})")

    tabs = st.tabs(
        ["🎯 Skill Level", "🗺️ Roadmap", "💡 Projects", "🎤 Interview Prep", "📄 Full Report"]
    )

    with tabs[0]:
        st.markdown(data["evaluation"])

    with tabs[1]:
        st.markdown(data["roadmap"])

    with tabs[2]:
        st.markdown(data["projects"])

    with tabs[3]:
        st.markdown(data["interview"])

    with tabs[4]:
        st.markdown(data["final_report"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "⬇️ Download Report (Markdown)",
            data=data["final_report"],
            file_name=f"career_report_{timestamp}.md",
            mime="text/markdown",
        )

    st.markdown("---")
    if st.button("🔄 Start Over with a New Goal"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()