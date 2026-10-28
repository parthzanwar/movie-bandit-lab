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

## Run the evaluation

```bash
.venv/bin/python run_evaluation.py
```

The evaluation uses seeds 0–4 and writes `results_reward.png` and
`results_regret.png` in this folder.
