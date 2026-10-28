# GenAI tutor: manual answer-quality review

**Review date:** 28 October 2026  
**Application:** Movie Bandit GenAI Lab at `http://127.0.0.1:8000`  
**Observed provider:** Apple Foundation Models (on device)

## Purpose and method

This is a small, manual end-to-end review of the tutor's answers. It complements
the separate 14-question retrieval benchmark; it does not replace that benchmark
or establish real-world recommendation accuracy.

I checked seven questions in the running application, including three
paraphrases. For explanation answers, I checked whether the response addressed the
question, agreed with the cited project passages, and avoided claims the
simulator cannot support. For exact-data questions, I checked the displayed
response against its read-only tool result and the simulator values. “Pass” means
adequate for this demonstration; “Partial” means useful but incomplete or
intermittent. This was one reviewer, not a blinded or statistically powered
study.

The live `/api/status` response currently reports `configured=true`, model
`Apple Foundation Models (on device)`, `key_present=false`, and 128 indexed
chunks. The answer-review note itself was added to the indexed sources after
the browser capture, so the already-open page retained its earlier header count
of 127. Explanation answers identified Apple Foundation Models and generated
cited prose. This confirms that a generative provider was configured and
invoked during the review. The Dangal and policy-comparison answers deliberately
use deterministic simulator tools, so they are not evidence of language-model
generation.

## Questions and outcomes

| # | Question | Mode and evidence | Review |
|---|---|---|---|
| 1 | “A reward of 1 means a like. How is that different from oracle p(like)?” | Apple Foundation Models answer with citations to the glossary and project walkthrough. The first answer correctly distinguished the sampled binary reward from an expected probability. A repeat request fell back to a retrieval-only response when the model did not return a usable answer. | **Partial** — the explanation was sound when generated, but the fallback shows provider reliability can vary. |
| 2 | “If a film has an 80 percent chance of a like but the simulator returns no-like, did the policy make an inaccurate recommendation?” *(paraphrase)* | The first model draft was only “No,” which was not an adequate explanation. A deterministic, cited source-summary path was then added for this specific stochastic-outcome misconception. The current user-facing response explains that an 80% event can still produce a no-like, and that one sampled outcome does not measure recommendation accuracy. | **Pass after correction** — the initial failure is retained here as a finding; the final response is clear and grounded in the glossary/walkthrough. |
| 3 | “What information does LinUCB use to choose among movies, and what does its uncertainty bonus do?” | Apple Foundation Models answer cited the glossary and walkthrough. It gave the uncertainty formula `alpha * sqrt(xᵀ A⁻¹ x)` and described uncertainty in the current context. | **Partial** — the response omitted the learned mean calculation `theta = A⁻¹b` and `mean = thetaᵀx`, although the cited passage contained it. |
| 4 | “Why might LinUCB give a less familiar movie a higher score than a movie it has chosen often?” *(paraphrase)* | Apple Foundation Models answer described the uncertainty bonus as encouraging trials with less evidence for the current context. The prompt was tightened after an earlier draft incorrectly suggested a large bonus meant a higher true like probability. | **Partial** — the latest answer avoids saying that the true like probability is higher, but the wording can still blur an under-explored movie action with a less-seen user. |
| 5 | “Before writing an explanation, how does the tutor find the project passages that should support its answer?” *(paraphrase)* | Apple Foundation Models answer cited project notes and correctly described searching passages before generation. | **Partial** — it did not explain the implemented TF-IDF vectors, cosine ranking, and top-five retrieval, so it was too generic for a full implementation explanation. |
| 6 | “Is Dangal a top oracle recommendation for a Rom-Com lover? Explain its genre match and where its popularity number comes from.” | Exact `inspect_movie_for_profile` simulator tool result; high-confidence tool response cites the project walkthrough. For MovieLens user #726, Dangal is Drama-only, genre match `0.167`, illustrative Bollywood popularity `0.960`, oracle `p(like) = 0.532`, and rank `20/42`, outside the top five. | **Pass** — the exact result corrects the unsupported claim that Dangal is a top Rom-Com recommendation and identifies the demo-data limitation. |
| 7 | “Compare all four policies for the Rom-com lover over 120 rounds: show total likes and average regret.” | Exact `compare_bandit_policies` result with seed 17, 120 rounds per policy, and a cited walkthrough passage. | **Pass** — the live UI matched the tool output: Random `56` likes / `0.433` average regret; Popularity `116` / `0.000`; Epsilon-Greedy `85` / `0.237`; LinUCB `78` / `0.392`. The response explains that these are stochastic simulated outcomes, not real-world accuracy. |

## Real interface evidence

The captured running UI showed the provider and index status in its header as:

> Apple Foundation Models (on device) · 127 indexed chunks

After the review document was added to the searchable source set, the current
`/api/status` value became 128 indexed chunks. The browser tab was already open,
so its displayed 127 is a pre-update snapshot; the provider itself matches.

The same live conversation displayed (1) a model-generated LinUCB explanation
with retrieved source cards, (2) the exact Dangal inspection with its Bollywood
demo-data warning, and (3) the exact four-policy comparison above. A live browser
capture of the application was made during this review; the exact messages and
values are transcribed here so the evidence is readable without relying on a
temporary browser screenshot.

The Dangal output matters because the project's movie list assigns Dangal only
the Drama genre, and its popularity is manually assigned. The simulator's
Rom-Com profile has strongest Comedy weight `0.667`, so the computed genre match
is only `0.167`. The tutor no longer presents this movie as a top-ranked choice.

## What this review supports

- A generative model was active and produced cited explanations in the live app.
- Exact profile/movie/policy values can be routed through the bounded tools and
  returned without asking the language model to invent them.
- The Dangal claim is now answered from the simulator data and is not supported
  as a top recommendation for the Rom-Com profile.
- The app still has answer-quality weaknesses: incomplete generated answers,
  occasional provider fallback, and wording that can be too general.
- The retrieval benchmark's Recall@K and MRR measure source retrieval, not the
  correctness or completeness of generated prose.
- No live human preferences are collected, so this project does not measure
  real-user recommendation accuracy.

## Next checks

For a stronger evaluation, review more held-out questions with a second
reviewer, include more paraphrases and misleading premises, and score factual
correctness, completeness, citation support, and provider availability
separately. Re-run the same prompts after any prompt or routing change and keep
the initial failures in the record rather than replacing them with only the
improved outputs.
