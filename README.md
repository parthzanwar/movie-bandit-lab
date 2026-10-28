# MovieLens contextual bandit demo

This project explores MovieLens 100K and compares four movie recommendation
policies: Random, Popularity, Epsilon-Greedy, and LinUCB. The dataset is in
`data/ml-100k`. Bollywood titles are hand-added demo items with illustrative
genre vectors and popularity scores; they do not have MovieLens ratings.

## Set up

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run the demo

```bash
.venv/bin/python -m streamlit run app.py
```

## Run the GenAI learning extension

The optional GenAI companion is a FastAPI + JavaScript app. It adds a local
RAG tutor over the project/course notes and safe, read-only tools for inspecting
the existing bandit simulation. The original reward math and policies stay the
same.

```bash
.venv/bin/python -m pip install -r requirements.txt -r requirements-genai.txt
cp -n .env.example .env
.venv/bin/python -m uvicorn genai_lab.api:app --reload
```

Then open <http://127.0.0.1:8000>. The default configuration uses Apple's
on-device Foundation Model on macOS 26 or later and compiles a small Swift
bridge on the first question. No API key or separate model download is needed.
The prompts, retrieved passages, and chat history stay on this Mac. If Apple
Foundation Models is unavailable, use a local Ollama OpenAI-compatible endpoint
or another compatible provider by setting `GENAI_PROVIDER=openai-compatible`,
`GENAI_BASE_URL`, and `GENAI_MODEL` in the private `.env` file. A hosted
provider also needs its API key there; hosted prompts leave the machine. Keep
`.env` private and never commit credentials. The `.env.example` file contains
no credentials. See `docs/GENAI_COURSE_ALIGNMENT.md` and
`docs/GENAI_MINI_PROJECT_REPORT.md` for course mapping, scope, evaluation, and
presentation notes.

Evaluate the retrieval component (no LLM call required):

```bash
.venv/bin/python -m genai_lab.evaluate_retrieval
```

## Run the evaluation

```bash
.venv/bin/python run_evaluation.py
```

The evaluation uses seeds 0–4 and writes `results_reward.png` and
`results_regret.png` in this folder.
