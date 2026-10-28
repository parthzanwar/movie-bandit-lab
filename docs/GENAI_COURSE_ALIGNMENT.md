# GenAI course alignment for Movie Bandit Lab

This extension applies a focused subset of the Generative AI course policy to
the existing movie recommendation simulator. It is a learning companion, not a
replacement for the bandit policies or a claim that the project has learned
real viewers' preferences.

## Course topics used

| Course topic | How this extension demonstrates it |
|---|---|
| Retrieval-Augmented Generation (RAG), CO3 | It retrieves relevant project and course-note chunks before asking a language model to answer. Every response can expose the retrieved source passages. |
| Text vectors and similarity, CO3 | A local `TfidfVectorizer` maps chunks and questions into sparse weighted term vectors. Their normalized dot product ranks candidate passages by cosine similarity. This is a lexical vector baseline, not a neural semantic embedding model. |
| Prompt templates, CO3 | The assistant uses an explicit system template that includes retrieved passages, the current viewer/policy settings, and citation rules. |
| Structured output parsing, CO3 | The model is asked for JSON fields (`answer`, `citations`, `confidence`, and `suggested_questions`); the server validates and sanitizes them before returning a response. |
| Agentic AI and tools, CO6 | A constrained intent router maps explicit data questions to one of five bounded, read-only functions: inspect a profile, list oracle recommendations, inspect one movie's fit for a profile, compare policies, or inspect a round. Only registered functions run, and exact values are formatted from their structured outputs. An optional OpenAI-compatible provider can also request a registered tool. |
| Quantitative evaluation, CO5 | A small manually labeled query set evaluates document retrieval with Recall@K and Mean Reciprocal Rank (MRR). |

## Deliberate scope choices

- The knowledge base is small, so the first implementation uses exact cosine
  search over an in-memory TF-IDF index. It avoids a separate database and
  approximate HNSW index until corpus size makes those useful.
- The current vectors capture word and phrase overlap. They can miss paraphrases.
  A future student-led extension could compare them with a pretrained sentence
  embedding model and a persistent vector database.
- On the development Mac, generation uses Apple's on-device Foundation Model
  through a small Swift bridge. The prompt, retrieved excerpts, and chat history
  remain on the device. An OpenAI-compatible provider is optional; if hosted,
  it receives those inputs. Never put a key in source control.
- Fine-tuning/LoRA/QLoRA is not used: the project has no suitable instruction
  dataset or justified task-specific training objective. Diffusion and large
  vision models are not used because this extension answers text questions and
  the existing project has no image-generation objective.
- The LLM explains and inspects the simulation. It does not choose actions for,
  update, or change the math of the four original policies.

## Course-policy source

Mukesh Patel School of Technology Management & Engineering, *Course Policy
Document: Generative AI, B.Tech (Artificial Intelligence), Semester VII,
Academic Year 2026-27*, supplied PDF, especially CO3/CO5/CO6 (pp. 2, 7-8),
RAG/vector retrieval and agent labs (pp. 6-8, 10-11), and GenAI project and
academic-integrity requirements (pp. 12-13).

The course policy says students must retain original implementation and
critical thinking, understand AI-assisted work, and cite references. Students
should adapt, explain, and validate this extension themselves and follow their
faculty's approval and disclosure rules before submitting it for assessment.
