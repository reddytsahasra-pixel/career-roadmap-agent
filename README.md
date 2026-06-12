# career-roadmap-agent


A multi-agent career coaching tool built with **AG2 (AutoGen)** and **Google Gemini**.
Given a target job role and salary package, it assesses your current skill level,
then generates a personalized learning roadmap, project suggestions, and an
interview preparation plan — all compiled into a final Markdown report.

## Pipeline

```
User (Target Role + Target Package)
    -> Assessment Agent        (asks skill-assessment questions)
    -> Skill Evaluation Agent  (Beginner / Intermediate / Advanced)
    -> Roadmap Agent           (personalized learning roadmap)
    -> Project Agent           (portfolio project suggestions)
    -> Interview Agent         (interview preparation plan)
    -> Final Report            (compiled Markdown document)
```

## Files

| File | Description |
|---|---|
| `career_pipeline_autogen.py` | Command-line version. Runs the full pipeline interactively in the terminal. |
| `app.py` | Streamlit web UI version. Same pipeline with a browser-based interface. |

## Requirements

- Python 3.10+ (tested on 3.13)
- A **Gemini API key** from [Google AI Studio](https://aistudio.google.com/api-keys) — **not** an OpenAI key

## Setup

### 1. Install dependencies

```bash
python -m pip install autogen-agentchat "autogen-ext[openai]" python-dotenv streamlit
```

> If `pip` or `streamlit` commands aren't recognized on Windows, use `python -m pip ...`
> and `python -m streamlit ...` instead — this calls them as modules and avoids PATH issues.

### 2. Set your Gemini API key

Create a file named `.env` in the same folder as the scripts:

```
GEMINI_API_KEY=your-gemini-key-here
```

Get a key from [aistudio.google.com/api-keys](https://aistudio.google.com/api-keys).

**Never commit `.env` to version control or share your key in screenshots/chat.**
If a key is ever exposed, delete it in AI Studio and create a new one.

## Usage

### Option A — Command line

```bash
python career_pipeline_autogen.py
```

You'll be prompted for a target role and package, then the assessment questions
will print to the terminal. Type your answers (one per line), then enter a blank
line to finish. The pipeline then runs through all stages and saves
`final_report_<timestamp>.md` in the current folder.

### Option B — Web UI (recommended)

```bash
python -m streamlit run app.py
```

This opens a browser tab. Enter your Gemini API key in the sidebar (or leave it
blank if `GEMINI_API_KEY` is already set via `.env`), set your target role/package,
and click through the assessment → results flow. The final report can be
downloaded as Markdown from the "Full Report" tab.

## Configuration

Both scripts default to the `gemini-2.5-flash-lite` model. To change it:

- **CLI script**: edit the `"model"` field in `config_list` / `model_client` near the top of `career_pipeline_autogen.py`
- **Streamlit app**: change the "Model" field in the sidebar

If you get a quota error (`limit: 0` for a model), try one of these alternatives:
- `gemini-2.0-flash-lite`
- `gemini-flash-lite-latest`
- `gemini-2.5-flash`

You can see which models your key has access to by running:

```python
import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'autogen'`**
The scripts use the newer AG2 API (`autogen_agentchat`, `autogen_ext`), not the
old `pyautogen` 0.2.x package (which requires Python <3.13). Make sure you
installed `autogen-agentchat` and `autogen-ext[openai]` as shown above.

**`429 ... TooManyRequests` / `quota exceeded`**
Your Gemini API key has hit its free-tier limit for that specific model. Try a
different model name (see Configuration above), or check current limits at
[ai.dev/rate-limit](https://ai.dev/rate-limit).

**`401 AuthenticationError: Incorrect API key`**
The key format doesn't match the provider. Gemini keys (from AI Studio) start
with `AQ.` and only work with the Gemini config in these scripts — they will
not work with OpenAI's API directly.

**`pip`/`streamlit` not recognized (Windows)**
Use `python -m pip ...` and `python -m streamlit ...` instead.

## Security Notes

- API keys are read from environment variables / `.env`, never hardcoded.
- If you ever paste a key into a chat, screenshot, or commit it to a repo,
  treat it as compromised and regenerate it immediately from AI Studio.
- The Streamlit app's API key field uses a masked password input and is never
  written to disk.
