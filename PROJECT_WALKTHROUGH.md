# Movie Bandit Lab: project walkthrough for presentation

## 1. Project in one minute

Movie Bandit Lab is an interactive demonstration of recommendation as a **multi-armed bandit** problem. At each round, a policy chooses one movie from a fixed slate of 42 titles. A simulated viewer either likes it (reward 1) or does not (reward 0). The policy sees that outcome and may update its strategy. The app compares four policies side by side and exposes the taste profile, expected like probabilities, learned model, and individual decisions.

This is an educational simulation, not a live recommender trained on real-time user feedback. Its rewards are generated from a frozen, hand-specified probability rule.

## 2. Data and preparation

The project uses MovieLens 100K (`data/ml-100k`). Ratings for each user are sorted chronologically and split 80%/20%; the training portion drives the demo. The test portion is not used by the interactive experiment.

The recommendation slate contains 42 movies:

- 30 MovieLens movies with at least 50 training ratings. The selection is greedy and aims to cover genres, using training rating count to break ties.
- 12 hand-added Bollywood demo titles. Their genre flags and popularity values are illustrative and are not MovieLens observations.

A user's context is a 19-element genre vector. It counts genres among their training ratings of 4 or 5, then normalizes the counts. The “Bollywood buff” is a synthetic profile formed from the Bollywood demo films; it is not a real MovieLens user.

## 3. How a simulated response is generated

For user context vector `x` and movie genre vector `g`, the system computes genre overlap `x · g`, combines it with the movie's frozen popularity score, and applies a logistic sigmoid:

```text
p(like) = sigmoid(4 × genre_overlap + 1 × popularity − 1.5)
```

A movie's displayed “Oracle p(like)” is this probability, not a guaranteed outcome. The environment samples a Bernoulli reward with that probability, so an individual round may produce either a like or no-like. MovieLens popularity is derived from training-set like counts and scaled to [0,1]. Bollywood popularity is manually assigned on the same scale.

## 4. Front-end controls and all combinations

There are four controls/dimensions:

| Control | Values | What it changes |
|---|---:|---|
| Simulated viewer | 11 profiles | Which taste vector the fixed environment uses |
| Policy | 4 policies | Which policy is highlighted and which policy's metrics/inspector are shown |
| Rounds per policy | 50 values: 10, 20, …, 500 | Number of decisions made by each policy in one run |
| Run the experiment | Run / not run | Starts the simulation; the chosen viewer and round count define the run |

The 11 profiles are Action junkie, Rom-com fan, Sci-fi dreamer, Drama devotee, Comedy fan, Mystery sleuth, Horror buff, Adventure seeker, Animation fan, War-story watcher, and Bollywood buff. The first ten are labels attached to distinct real MovieLens training users selected for genre alignment; they are not artificially edited tastes. Bollywood buff is the synthetic profile described above.

The 4 policy choices are Random, Popularity, Epsilon-Greedy, and LinUCB. Thus there are **11 × 4 × 50 = 2,200 selectable viewer/policy/round-count combinations**. Run/not-run is an action, not another experiment setting. Importantly, selecting a policy does **not** cause only that policy to run: every experiment executes all four policies for the chosen viewer and round count. The policy selection changes the highlighted curve, selected-policy metric cards, round-by-round inspector, and current policy context in the view. The experiment run itself is keyed to viewer and round count; changing only the display policy does not require new policy data.

### How to interpret the displayed sections

- **Movie slate / Liked genres:** fixed slate size (42) and number of non-zero genres in the selected viewer's profile.
- **Likes · selected policy:** observed simulated reward total for the selected policy, after a matching experiment has run.
- **Average regret:** average per-round difference between the best movie's oracle probability for this viewer and the chosen movie's oracle probability. Lower means choices were closer to the oracle best in expectation; this is not the same as number of observed dislikes.
- **Reward over time:** cumulative likes for all four policies. The selected policy is emphasized; the others remain as comparisons.
- **This viewer's taste:** normalized genre weights used as the user's context.
- **Most likely to land:** the five highest frozen oracle probabilities for this viewer, before the experiment. These are expected probabilities, not outcomes.
- **What LinUCB learned:** its learned genre coefficients, estimated reward, uncertainty bonus, and current highest upper-confidence choice. This panel exists even when another policy is selected.
- **Round-by-round inspector:** the selected policy's chosen movie, exploration/exploitation label, oracle probability, and sampled reward. For LinUCB, it also shows the estimate + bonus = score for that round.
- **Policy scorecard:** reward and average regret for all four policies after a matching run.

If the selected viewer or round count changes after a run, the result no longer matches that configuration, so result-dependent displays ask for a new run. Selecting another display policy simply changes which policy's details are inspected.

## 5. What the four policies do

### Random
Chooses uniformly at random from all 42 movies every round. It ignores both the user's context and prior rewards. This is a blind baseline.

### Popularity
Always chooses the movie with the largest training-based like-rate estimate. It does not adapt to the selected user or to rewards observed in the run. In this app the estimate uses candidate training like rates, with the frozen popularity score as fallback.

### Epsilon-Greedy
Maintains a running average reward separately for each movie, ignoring viewer context. With epsilon = 0.1, it explores by selecting a random movie about 10% of rounds; otherwise it selects the movie with the best current average. Early on, all averages start at zero, so tie-breaking favors the first index. Its observed reward updates that movie's sample average.

### LinUCB
Maintains a separate linear reward estimate for each movie using the 19-dimensional viewer context. It selects the movie with the highest learned mean reward plus an uncertainty bonus. The bonus encourages trying actions with less evidence for the current context. Here alpha = 1.0. After the reward, it updates only the selected movie's model. Popularity is part of the environment's oracle probability, but is not a direct LinUCB feature.

## 6. What happens when Run is pressed

1. The app creates a separate environment and policy instance for each of the four strategies.
2. For each round, each strategy sees the same selected viewer profile, selects a movie, and receives a reward sampled from that movie's fixed probability. Each policy has its own seeded random stream; environments are independent.
3. Each strategy updates from its own outcome (Random and Popularity do not learn; Epsilon-Greedy and LinUCB do).
4. The app records chosen titles, probabilities, rewards, exploration labels, cumulative rewards, total rewards, and mean regret.
5. The interface shows comparisons and detail views for the chosen viewer and round count.

The interactive run uses a fixed seed (17 plus policy offsets), so the same configuration is reproducible. The four policy streams are independent, not paired to receive the exact same reward draws. Consequently, comparisons are informative demonstrations, while repeated seeds or the separate evaluation script are better for robust statistical claims.

## 7. How to demonstrate the options to a professor

A concise walkthrough:

1. Choose a genre-labeled profile, such as **Horror buff**, and explain that it maps to a real training user selected for genre alignment.
2. Choose **LinUCB** as the display policy and set **120 rounds**. Explain that this setting highlights its choices, while all four policies will still run.
3. Before running, point out the taste chart and the top-five oracle recommendations. Explain that probabilities describe the simulator's expected response.
4. Press **Run the experiment**. Compare the cumulative reward lines; then read the selected-policy reward and average regret.
5. Move the round inspector to an early and a late round. For LinUCB, show how the estimate and uncertainty bonus combine.
6. Change the display policy to Epsilon-Greedy and compare its choices. It tracks averages per movie and ignores the viewer's genre context.
7. Change viewer or round count, rerun, and discuss how personalization or a longer learning horizon can affect results. Optionally select Bollywood buff and clearly explain that this is synthetic demo data.

Useful questions to discuss: Does personalization help over popularity? How does exploration affect early versus later rewards? How does additional interaction change uncertainty? Why can a policy with strong expected probabilities still observe a no-like on one round?

## 8. Limits and interpretation

- Rewards are simulated from a manually specified logistic rule, not directly measured as new user reactions.
- The genre-alignment profile names are explanatory labels, not user identities or controlled experimental treatments.
- The 12 Bollywood films have illustrative genres and popularity scores and no MovieLens ratings.
- The fixed seed makes a run repeatable but does not establish that one strategy is statistically superior. Use multiple seeds and aggregate results for performance claims.
- Average regret uses the known oracle probabilities, available because this is a simulation. A real recommender would not know the true response probabilities.
- The separate evaluation script is distinct from the interactive front end. Its results and plots should be described with their own run settings and seed aggregation.

## 9. Project files

- `app.py`: interactive front end, controls, charts, explanations, experiment loop, and inspector.
- `environment.py`: MovieLens loading, train/test split, candidate slate, user/movie genre vectors, popularity scores, synthetic profile, and reward environment.
- `agents.py`: implementations of the four policies.
- `run_evaluation.py`: multi-seed offline policy evaluation and plots.
- `data/ml-100k/`: MovieLens 100K source data.
- `results_reward.png`, `results_regret.png`: evaluation chart outputs when regenerated by the evaluation script.
- `genai_lab/`: optional FastAPI + JavaScript RAG tutor and read-only simulation tools.
- `docs/GENAI_COURSE_ALIGNMENT.md`: course outcomes and topic mapping for the extension.
- `docs/GENAI_MINI_PROJECT_REPORT.md`: mini-project scope, 10-mark rubric, evaluation, limitations, and viva walkthrough.

## 10. Short oral summary

“Movie Bandit Lab compares four strategies for choosing movies under uncertain feedback. Each viewer is represented by a genre profile derived from training ratings, and each movie has genre and popularity features. A fixed logistic oracle converts their match into a like probability, then samples a binary reward. Random and popularity are baselines; Epsilon-Greedy learns per-movie averages; LinUCB learns context-dependent reward estimates and explores using uncertainty. The front end lets us choose one of 11 viewer profiles, a display policy, and 10–500 rounds in steps of 10. Every run evaluates all four policies, making it possible to compare cumulative likes, regret, learned parameters, and individual decisions. The result is an inspectable teaching simulation, with clear limits because feedback and some movie metadata are simulated.”

## 11. Optional GenAI course extension

The project also includes a separate **Movie Bandit GenAI Lab** served by
FastAPI and JavaScript. It adds a course tutor over the project and course
notes; the original Streamlit simulation, policy code, reward formula, and seed
`17` remain the bandit experiment.

The tutor uses a small Retrieval-Augmented Generation (RAG) pipeline. It splits
the source files into overlapping chunks, represents them with local TF-IDF
vectors, ranks them with cosine similarity, and gives the strongest passages
to a configured language model. Answers are requested as JSON and validated
before display, with retrieved source passages shown in the interface. This is
a transparent lexical retrieval baseline, not a neural semantic embedding
model.

When a language model is configured, a bounded agent can call five read-only
tools: inspect a viewer profile, list the simulator's oracle top movies, inspect
one movie's fit for a viewer, compare the four policies, or inspect a particular
round. Movie-fit inspection reports genre overlap, oracle probability, rank, and
whether popularity comes from MovieLens training data or illustrative
Bollywood-demo metadata. These tools use the existing environment and policy
classes; they cannot change project files or policy state. On the 14-question
manually labeled retrieval set, current retrieval measures Recall@1 `0.571`,
Recall@3 `0.929`, Recall@5 `1.000`, and MRR `0.752`. A separate seven-question
manual answer review confirmed cited model-generated explanations and exact
tool results, while also recording incomplete answers, a first-draft
unsupported claim that was corrected, and an intermittent provider fallback. See
[`docs/GENAI_ANSWER_QUALITY_EVALUATION.md`](docs/GENAI_ANSWER_QUALITY_EVALUATION.md)
for the actual prompts and review outcomes. The live development environment
reported Apple Foundation Models on device as its active provider. Neither
evaluation establishes real-user recommendation accuracy.

The extension demonstrates course topics in RAG, text-vector similarity,
prompt templates, structured output parsing, tool-using agents, and quantitative
evaluation. It does not fine-tune an LLM or add diffusion/vision models because
those techniques do not fit this text-focused project or its available data.
The current development setup uses Apple Foundation Models on device; other
machines may show retrieval-only mode until a supported provider is configured.
See `docs/GENAI_COURSE_ALIGNMENT.md`
and `docs/GENAI_MINI_PROJECT_REPORT.md` for course mapping, setup, the 10-mark
rubric, evaluation method, and presentation outline.
