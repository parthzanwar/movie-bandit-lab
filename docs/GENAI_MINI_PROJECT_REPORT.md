# Grounded GenAI Tutor for Movie Bandit Lab

## Mini-project scope and 10-mark rubric

**Proposed title:** *A Retrieval-Grounded GenAI Tutor with Read-Only Agent Tools
for a Contextual Movie Bandit Simulator*.

The course policy asks groups to get the mini-project topic approved by faculty.
Confirm this title and the chosen model/provider before submitting or presenting.
This extension is deliberately bounded: the existing recommendation policies
remain the action-taking algorithms, while GenAI explains project material and
can inspect a reproducible simulation through a small set of read-only tools.

| Course rubric | Evidence in this project | Marks |
|---|---|---:|
| Domain knowledge and literature review | Problem statement, course outcome mapping, RAG/TF-IDF/agent design, references, and limits | 2 |
| Implementation, performance evaluation, learning beyond class | FastAPI + JavaScript UI, retrieval pipeline, prompt template, structured parsing, tool agent, retrieval metrics, and reproducible experiment procedure | 5 |
| Report | This report, architecture, equations, rubric map, evaluation method/results, limitations, references, and demo guide | 3 |
| **Total** |  | **10** |

This is a report scaffold and implementation guide, not a substitute for the
student's own explanation, instructor approval, or required contribution.
Review the code, run the system, add your own observations, and follow the
course's GenAI-use disclosure and citation requirements.

## 1. Problem and motivation

Movie Bandit Lab already compares four policies that recommend one movie from a
42-title action slate. A student presenting it may need to explain an oracle
probability, a policy's exploration choice, cumulative regret, and what parts of
the system use measured MovieLens data versus simulation. A generic chatbot can
make unsupported claims about the repository. This extension grounds answers
in project files and course notes, displays the retrieved passages, and may
call safe tools to inspect current simulation settings.

**Research question:** Does retrieving project-specific evidence and allowing
bounded read-only simulation tools make a GenAI tutor more useful and verifiable
for explaining a contextual bandit project than an unconstrained prompt alone?

The prototype focuses on retrieval quality and transparent evidence. A small
manual answer-quality spot check now complements the retrieval benchmark; it is
not a formal user study and does not prove that every generated explanation is
true or that students learn more.

## 2. System architecture

```mermaid
flowchart LR
  U[Student asks a question] --> API[FastAPI request validation]
  API --> R[Chunk project docs and course-alignment notes]
  R --> V[TF-IDF sparse vectors]
  V --> S[Exact cosine top-5 retrieval]
  S --> P[Grounded prompt with source IDs]
  P --> L[Apple on-device Foundation Model]
  S --> G[Constrained intent router]
  G --> T[Allow-listed read-only bandit tools]
  T --> B[Existing environment.py and agents.py]
  B --> F[Exact result formatter]
  L --> O[JSON output parser and citation validation]
  O --> UI[JavaScript UI: answer and evidence]
  F --> UI
```

The new web app is served with FastAPI and plain JavaScript. The browser sends
the question, selected profile, policy, and horizon to `/api/ask`. The server
retrieves local evidence, then either constructs a grounded prompt for Apple's
on-device Foundation Model through a Swift bridge or routes an explicit request
for current simulator data to a bounded read-only tool. Generated answers are
parsed and their citations validated; exact tool values are formatted directly
from the simulator output.
The existing `app.py`, `environment.py`, `agents.py`, reward rule, and seed
schedule are not changed by this extension.

## 3. Retrieval and prompt design

### Knowledge base and chunking

The index reads `README.md`, `PROJECT_WALKTHROUGH.md`,
`docs/GENAI_GLOSSARY.md`, `agents.py`, `environment.py`, `run_evaluation.py`,
`app.py`, `docs/GENAI_COURSE_ALIGNMENT.md`, and
`docs/GENAI_MINI_PROJECT_REPORT.md`. Text is split into overlapping windows of
about 1,100 characters with a 180-character overlap. Each retrieved passage
keeps its repository path and chunk number, which the UI exposes as evidence.

### TF-IDF vectors and ranking

For term `t` in chunk `d`, the vectorizer weights term frequency by inverse
document frequency:

```text
tf-idf(t,d) = tf(t,d) × idf(t)
idf(t) = log((1 + N) / (1 + df(t))) + 1
```

Each row is L2-normalized. A query is transformed with the same vocabulary,
then ranked with cosine similarity. For normalized vectors, cosine similarity
is their dot product. The top five passages are inserted into the prompt.
Common novice terms such as “reward”, “like”, “regret”, and “accuracy” receive
a short deterministic concept expansion before vectorization to improve
matching against the project's exact glossary definitions.
This is a sparse, lexical retrieval baseline; it is not a neural semantic
embedding model. Exact search is suitable for this small repository corpus. A
future comparison can add dense embeddings, ChromaDB/Weaviate, and HNSW.

### Prompt template and grounded behavior

The system prompt says the simulator is synthetic, requires evidence IDs for
repository-specific claims, forbids claiming to know a real viewer's current
preferences, and asks the model to state when retrieved evidence is insufficient.
Retrieved passages are treated as reference data rather than instructions.
The current viewer, policy, and horizon are supplied as context, but the model
must use a tool to obtain exact run-dependent values.

## 4. Bounded agent tools and output parser

The agent layer exposes only these five registered functions. By default, a
constrained intent router selects a tool for explicit questions about live
values; the formatter copies exact values from the tool result instead of
asking the language model to invent or estimate them. An optional
OpenAI-compatible provider can also request a registered tool for other
questions.

1. `inspect_viewer_profile`: returns the selected profile's top genre weights
   and whether it is synthetic.
2. `list_oracle_recommendations`: returns top titles and frozen oracle
   probabilities.
3. `inspect_movie_for_profile`: returns one candidate movie's genre match,
   popularity basis, oracle probability, and rank for the selected viewer. It
   can distinguish MovieLens-derived metadata from hand-assigned Bollywood
   demo values.
4. `compare_bandit_policies`: runs the existing four policies with seed 17 for
   a selected viewer/horizon and returns sampled likes and average regret.
5. `inspect_policy_decision`: returns a selected round's movie, oracle
   probability, sampled reward, and exploration label.

The server validates profile/policy names, constrains the horizon to 10–500
rounds, and returns plain data. The assistant cannot execute arbitrary code,
change files, or update policies.

The model is asked to return a JSON object with `answer`, `citations`,
`confidence`, and `suggested_questions`. Pydantic validates the shape; the
server drops citations that do not match retrieved source IDs. The frontend
renders generated text with DOM `textContent`, not unsanitized HTML.

## 5. Quantitative evaluation

### Retrieval benchmark

The local evaluation set contains 14 manually written questions with relevant
source files labeled in `genai_lab/evaluation_queries.json`. It is a small
course-project benchmark, not a standardized external benchmark. Run it with:

```bash
.venv/bin/python -m genai_lab.evaluate_retrieval
```

Metrics:

- **Recall@K (source-level):** fraction of questions for which at least one
  labeled relevant source appears among the top K retrieved chunks.
- **MRR:** average reciprocal rank of the first chunk from any labeled relevant
  source; an earlier relevant chunk scores higher.

| Metric | Current result |
|---|---:|
| Recall@1 | 0.571 |
| Recall@3 | 0.929 |
| Recall@5 | 1.000 |
| MRR | 0.752 |

These are measured on the 14-query, source-labeled prototype set in the current
workspace. Recall@3 means at least one labeled relevant source appeared in the
top three results for 13 of 14 queries; Recall@5 did so for all 14 queries.
Because the set is small and manually labeled, report these as initial
engineering measurements, not generalizable retrieval quality.

These metrics evaluate retrieval only, not LLM factuality, citation precision,
user learning, or recommendation accuracy. A separate seven-question manual
answer review found both supported answers and quality failures; see
[`GENAI_ANSWER_QUALITY_EVALUATION.md`](GENAI_ANSWER_QUALITY_EVALUATION.md) for
the prompts, outcomes, and caveats. The original bandit evaluation is a
separate simulation using five random seeds and 2,000 rounds per policy/seed;
it evaluates policy reward and model-relative pseudo-regret, not real-user
preference accuracy.

### Suggested follow-up experiments

1. Compare chunk sizes (700, 1,100, and 1,500 characters) and overlap (0, 100,
   and 180 characters) using the same held-out query set.
2. Compare TF-IDF retrieval against a pretrained sentence-embedding model using
   the same labels, Recall@K, MRR, latency, and memory.
3. Conduct a small instructor-approved user study that rates answer correctness,
   evidence usefulness, and explanation clarity. Do not claim results until it
   is actually run and approved.
4. Expand the answer-quality review with more held-out paraphrases and track
   factual support, completeness, and citation precision separately.

## 6. Running the extension

Install project and GenAI web dependencies inside the project virtual
environment, then copy `.env.example` to `.env`:

```bash
.venv/bin/python -m pip install -r requirements.txt -r requirements-genai.txt
cp -n .env.example .env
.venv/bin/python -m uvicorn genai_lab.api:app --reload
```

Open `http://127.0.0.1:8000`. Apple Foundation Models on device was active in
the current development setup; the live status endpoint and interface header
reported that provider during the 28 October 2026 review. The Swift bridge is
compiled into a temporary cache on first use. Prompts, chat history, and
retrieved excerpts stay local. For other machines, an OpenAI-compatible local
or hosted provider can be selected in `.env`; hosted providers receive those
inputs. Do not commit `.env` or API keys.

## 7. Limitations and responsible use

- TF-IDF favors literal term overlap and may miss semantic paraphrases.
- The 14-question labeled set is small and was authored for this codebase; the
  measurements need a larger held-out set for stronger claims.
- The answer-quality check is a small manual spot check. Unsupported claims,
  incomplete answers, and intermittent provider fallback still need attention.
- Bandit rewards are generated from a hand-specified logistic oracle. The agent
  sees simulator outputs, not a real viewer's preferences.
- “Best” policy can vary by seed; a single interactive run is a demonstration.
- The extension does not use fine-tuning/LoRA, diffusion, or vision models
  because no supported training corpus or image-generation objective exists.
- Follow the course policy: get the project topic approved; understand and
  contribute original work; cite sources and disclose AI assistance as required.

## 8. Demo / viva walkthrough

1. Select **Horror buff**, **LinUCB**, and **120 rounds**. Point out the profile
   summary and the fixed simulator's top oracle options.
2. Ask “Explain regret gap in simple terms.” Show the retrieved walkthrough
   evidence and distinguish expected probability gap from the sampled like.
3. Ask “Why did the selected policy choose its movie in the final round?” Show
   the read-only decision tool's JSON and explain the estimate/bonus/score
   information for LinUCB.
4. Ask for a four-policy comparison. Explain that observed likes are one
   stochastic run and average regret measures closeness to the simulator oracle.
5. Ask how the assistant uses RAG. Trace question → chunk retrieval → TF-IDF
   cosine ranking → grounded prompt → structured answer with citations.
6. Show the evaluation script's Recall@K and MRR and state what they evaluate
   and what they do not.
7. Show the separate answer-quality spot check. Explain that it reviews a few
   actual model/tool responses and is not a broad factuality benchmark.

## References

1. Mukesh Patel School of Technology Management & Engineering. *Course Policy
   Document: Generative AI*, B.Tech (Artificial Intelligence), Semester VII,
   Academic Year 2026-27, supplied course-policy PDF, pp. 2, 6-13.
2. MovieLens 100K README and project source files included in this repository.
3. scikit-learn Developers. “Feature extraction” and `TfidfVectorizer`
   documentation: https://scikit-learn.org/stable/modules/feature_extraction.html
4. FastAPI documentation. “Static Files”: https://fastapi.tiangolo.com/tutorial/static-files/
5. OpenAI. “Chat Completions API Reference” (tool calling and response
   formats): https://platform.openai.com/docs/api-reference/chat
6. Ollama. “OpenAI Compatibility” (optional local-compatible endpoint):
   https://github.com/ollama/ollama/blob/main/docs/api/openai-compatibility.mdx
