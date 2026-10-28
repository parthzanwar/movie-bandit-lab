"""Presentation view for the MovieLens contextual bandit demo."""

from pathlib import Path
from html import escape
import re
from textwrap import dedent

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

from agents import EpsilonGreedy, LinUCB, PopularityPolicy, RandomPolicy
from environment import (
    BOLLYWOOD_DEMO_ITEMS,
    BOLLYWOOD_ITEM_IDS,
    BOLLYWOOD_USER_ID,
    DATA_DIR,
    GENRE_NAMES,
    MOVIE_COLUMNS,
    MovieBanditEnvironment,
    candidate_movies,
    demo_user_contexts,
    movie_genre_vectors,
    movie_popularity_score,
    movie_titles,
    train,
    user_contexts,
)


st.set_page_config(
    page_title="Movie Bandit Lab",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#090D16"
PANEL = "rgba(17, 24, 39, 0.75)"
MUTED = "#9EADB8"
TEXT = "#E8EDF0"
SAFFRON = "#F59E0B"
CYAN = "#22D3EE"
PALE = "#CAD4DA"
POLICY_COLORS = {
    "Random": "#64748B",
    "Popularity": "#10B981",
    "Epsilon-Greedy": "#3B82F6",
    "LinUCB": "#F59E0B",
}

st.markdown(
    dedent("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
    :root { color-scheme: dark; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    body, [data-testid="stMetricValue"], .kpi-value, .leader-value, .mono-tag, .tiny-pill { font-variant-numeric: tabular-nums; }
    h1, h2, h3, h4 { font-family: 'Plus Jakarta Sans', sans-serif; letter-spacing: -0.035em; }
    .stApp { background: #090D16; }
    [data-testid="stSidebar"] { background: #06090F; border-right: 1px solid rgba(255,255,255,.08); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.3rem; }
    .hero-kicker { color: #22D3EE; font: 700 .7rem 'JetBrains Mono', monospace; letter-spacing: .12em; margin-bottom: .65rem; }
    .hero-title { font: 800 clamp(2.2rem,4vw,3.6rem) 'Plus Jakarta Sans', sans-serif; letter-spacing:-.055em; line-height:1.05; margin:.3rem 0 .8rem; color:#F8FAFC; }
    .hero-copy { color: #9AA8BB; font-size: 1.04rem; max-width: 54rem; margin: 0 0 1rem; line-height:1.7; }
    .hero-rule { border: 0; border-top: 1px solid rgba(255,255,255,.08); margin: 1.25rem 0; }
    .section-note { color: #9EADB8; font-size: .92rem; }
    .stButton > button[kind="primary"] { background: #F59E0B; border: 0; color: #151A1E; font-weight: 800; border-radius:9px; }
    .stButton > button[kind="primary"]:hover { background: #FBBF24; color: #151A1E; }
    [data-testid="stMetricValue"] { color: #F59E0B; font: 600 1.7rem 'JetBrains Mono',monospace; font-variant-numeric:tabular-nums; }
    [data-testid="stMetricLabel"] { color: #AAB7C0; }
    div[data-testid="stExpander"] { border-color: rgba(255,255,255,.08); background: rgba(17,24,39,.5); border-radius:14px; }
    .bento-card,.kpi-card,.movie-row-card,.decision-card,.leader-card { background:rgba(17,24,39,.75); backdrop-filter:blur(12px); border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:20px; transition:border-color .2s,transform .2s; position:relative; overflow:hidden; }
    .bento-card:before,.kpi-card:before { content:''; position:absolute;left:0;right:0;top:0;height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.22),transparent); }
    .bento-card:hover,.kpi-card:hover,.movie-row-card:hover,.leader-card:hover { border-color:rgba(242,184,75,.35); transform:translateY(-1px); }
    .kpi-card { min-height:125px; padding:17px 18px; }
    .kpi-label { color:#94A3B8;font:700 .68rem 'Inter',sans-serif;letter-spacing:.12em;text-transform:uppercase; }
    .kpi-value { color:#F8FAFC;font:600 1.75rem 'JetBrains Mono',monospace;font-variant-numeric:tabular-nums;margin:10px 0 4px; }
    .kpi-sub { color:#64748B;font-size:.75rem;line-height:1.4; }
    .hero-status,.tiny-pill,.badge-explore,.badge-exploit,.badge-liked,.badge-disliked,.year-pill,.source-pill { display:inline-flex;align-items:center;border-radius:999px;padding:5px 9px;font:600 .68rem 'JetBrains Mono',monospace;letter-spacing:.02em; }
    .hero-status { color:#67E8F9;background:rgba(34,211,238,.08);border:1px solid rgba(34,211,238,.18); }
    .walkthrough { display:flex;flex-wrap:wrap;gap:9px;margin:18px 0 4px; }
    .tiny-pill { color:#CBD5E1;background:rgba(148,163,184,.08);border:1px solid rgba(148,163,184,.14); }
    .badge-explore { color:#FCD34D;background:rgba(245,158,11,.13);box-shadow:0 0 18px rgba(245,158,11,.12); }
    .badge-exploit { color:#67E8F9;background:rgba(34,211,238,.12);box-shadow:0 0 18px rgba(34,211,238,.10); }
    .badge-liked { color:#6EE7B7;background:rgba(16,185,129,.15); }
    .badge-disliked { color:#FDA4AF;background:rgba(244,114,114,.12); }
    .movie-row-card { padding:13px 15px;margin:9px 0; }
    .movie-top { display:flex;justify-content:space-between;gap:10px;align-items:center; }
    .movie-title { font:700 .9rem 'Plus Jakarta Sans',sans-serif;color:#F1F5F9; }
    .rank { color:#F59E0B;font:600 .75rem 'JetBrains Mono',monospace;margin-right:9px; }
    .year-pill,.source-pill { color:#94A3B8;background:rgba(148,163,184,.1);font-size:.62rem;padding:3px 7px; }
    .prob-line { display:flex;justify-content:space-between;color:#94A3B8;font:.67rem 'JetBrains Mono',monospace;margin:10px 0 5px; }
    .prob-track,.split-track,.regret-track { background:#1E293B;border-radius:99px;height:6px;overflow:hidden; }
    .prob-fill { height:100%;background:linear-gradient(90deg,#10B981,#22D3EE);border-radius:99px; }
    .movie-meta { display:flex;gap:7px;flex-wrap:wrap;margin-top:9px; }
    .mono-tag { color:#CBD5E1;background:rgba(148,163,184,.08);border:1px solid rgba(148,163,184,.12);border-radius:6px;padding:4px 7px;font:500 .64rem 'JetBrains Mono',monospace; }
    .decision-card { display:grid;grid-template-columns:1.15fr 1fr .65fr;gap:18px;padding:22px; }
    .decision-title { font:800 1.35rem 'Plus Jakarta Sans',sans-serif;color:#F8FAFC;margin:12px 0 7px; }
    .decision-label { color:#64748B;font:700 .63rem 'Inter',sans-serif;letter-spacing:.12em;text-transform:uppercase; }
    .genre-tags { display:flex;gap:5px;flex-wrap:wrap;margin-top:12px; }
    .telemetry { display:grid;gap:9px;margin:13px 0; }
    .equation { display:flex;gap:7px;flex-wrap:wrap;align-items:center; }
    .eq-block { background:#0B1220;border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:8px 10px;color:#CBD5E1;font:500 .7rem 'JetBrains Mono',monospace; }
    .mean-fill { height:100%;background:#22D3EE;float:left; }.bonus-fill { height:100%;background:#F59E0B;float:left; }
    .leader-card { padding:17px;min-height:190px; }
    .leader-policy { font:700 .9rem 'Plus Jakarta Sans',sans-serif;color:#E2E8F0; }
    .leader-value { font:600 1.45rem 'JetBrains Mono',monospace;color:#F8FAFC;margin:15px 0 4px;font-variant-numeric:tabular-nums; }
    .best-badge { color:#FCD34D;font:700 .59rem 'JetBrains Mono',monospace;letter-spacing:.06em;margin-top:8px; }
    .footer-note { border-top: 1px solid rgba(255,255,255,.08); color: #64748B; font-size: .83rem; padding-top: 1rem; margin-top: 2.1rem; }
    @media(max-width:800px){.decision-card{grid-template-columns:1fr}.walkthrough{gap:6px}}
    </style>
    """),
    unsafe_allow_html=True,
)


POLICY_NAMES = ["Random", "Popularity", "Epsilon-Greedy", "LinUCB"]
PROFILE_TARGETS = [
    ("Action junkie", ["Action"]),
    ("Rom-com fan", ["Romance", "Comedy"]),
    ("Sci-fi dreamer", ["Sci-Fi"]),
    ("Drama devotee", ["Drama"]),
    ("Comedy fan", ["Comedy"]),
    ("Mystery sleuth", ["Mystery"]),
    ("Horror buff", ["Horror"]),
    ("Adventure seeker", ["Adventure"]),
    ("Animation fan", ["Animation"]),
    ("War-story watcher", ["War"]),
    ("Bollywood buff", None),
]


@st.cache_data(show_spinner=False)
def load_movie_data(data_dir):
    """Cache the MovieLens tables used to label the fixed movie slate."""
    data_dir = Path(data_dir)
    ratings = pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    movies = pd.read_csv(
        data_dir / "u.item",
        sep="|",
        names=MOVIE_COLUMNS,
        encoding="latin-1",
    )
    return ratings, movies


def build_profile_users():
    """Choose distinct real training profiles close to each playful label."""
    ordered_users = sorted(user_contexts)
    used_users = set()
    chosen_users = {}
    for label, target_genres in PROFILE_TARGETS:
        if target_genres is None:
            chosen_users[label] = BOLLYWOOD_USER_ID
            continue
        target = np.zeros(len(GENRE_NAMES), dtype=float)
        for genre in target_genres:
            target[GENRE_NAMES.index(genre)] = 1.0 / len(target_genres)
        ranked = sorted(
            ordered_users,
            key=lambda uid: float(user_contexts[uid] @ target),
            reverse=True,
        )
        selected = next((uid for uid in ranked if uid not in used_users), ranked[0])
        chosen_users[label] = selected
        used_users.add(selected)
    return chosen_users


PROFILE_USERS = build_profile_users()
candidate_train = train[train.item_id.isin(candidate_movies)]
candidate_train_like_rates = (
    candidate_train.assign(liked=(candidate_train.rating >= 4).astype(int))
    .groupby("item_id").liked.mean()
)
popularity_rates = np.array(
    [
        candidate_train_like_rates.get(item_id, movie_popularity_score[item_id])
        for item_id in candidate_movies
    ],
    dtype=float,
)


def make_policy(name, seed):
    if name == "Random":
        return RandomPolicy(len(candidate_movies), random_state=seed)
    if name == "Popularity":
        return PopularityPolicy(popularity_rates)
    if name == "Epsilon-Greedy":
        return EpsilonGreedy(len(candidate_movies), epsilon=0.1, random_state=seed)
    return LinUCB(len(candidate_movies), context_size=len(GENRE_NAMES), alpha=1.0)


def linucb_scores(policy, context):
    """Return LinUCB's fitted reward estimates and uncertainty for each movie."""
    context = np.asarray(context, dtype=float)
    means = np.empty(policy.n_arms, dtype=float)
    bonuses = np.empty(policy.n_arms, dtype=float)
    for arm in range(policy.n_arms):
        inverse = np.linalg.inv(policy.A[arm])
        theta = inverse @ policy.b[arm]
        means[arm] = theta @ context
        bonuses[arm] = policy.alpha * np.sqrt(max(0.0, context @ inverse @ context))
    return means, bonuses, means + bonuses


def run_demo(user_id, n_rounds, seed, progress_bar):
    """Run all policies against the same fixed user and record each decision."""
    environments = {
        name: MovieBanditEnvironment(random_state=seed + index)
        for index, name in enumerate(POLICY_NAMES)
    }
    policies = {
        name: make_policy(name, seed + index)
        for index, name in enumerate(POLICY_NAMES)
    }
    oracle_probabilities = environments["Random"].predict_probabilities(user_id)
    best_probability = float(oracle_probabilities.max())
    cumulative_rewards = {name: [] for name in POLICY_NAMES}
    cumulative_regrets = {name: [] for name in POLICY_NAMES}
    reward_totals = {name: 0 for name in POLICY_NAMES}
    regret_totals = {name: 0.0 for name in POLICY_NAMES}
    decisions = {name: [] for name in POLICY_NAMES}

    for round_index in range(n_rounds):
        for name in POLICY_NAMES:
            environment = environments[name]
            policy = policies[name]
            context = environment.reset(user_id)
            means = bonuses = scores = None
            greedy_arm = None
            if name == "LinUCB":
                means, bonuses, scores = linucb_scores(policy, context)
                greedy_arm = int(np.argmax(means))

            action = policy.select_action(context)
            reward = environment.step(action)
            if name == "Random":
                behavior = "Exploration"
            elif name == "Popularity":
                behavior = "Exploitation"
            elif name == "Epsilon-Greedy":
                behavior = "Exploration" if policy.last_exploring else "Exploitation"
            else:
                # The bonus is always scored; it counts as exploration when it
                # changes the chosen arm from LinUCB's greedy reward estimate.
                behavior = "Exploration" if action != greedy_arm else "Exploitation"

            decisions[name].append(
                {
                    "round": round_index + 1,
                    "movie_id": int(candidate_movies[action]),
                    "movie": movie_titles[int(candidate_movies[action])],
                    "oracle_p_like": float(oracle_probabilities[action]),
                    "reward": int(reward),
                    "behavior": behavior,
                    "linucb_estimate": float(means[action]) if means is not None else None,
                    "ucb_bonus": float(bonuses[action]) if bonuses is not None else None,
                    "ucb_score": float(scores[action]) if scores is not None else None,
                }
            )
            policy.update(action, context, reward)
            reward_totals[name] += int(reward)
            regret_totals[name] += best_probability - oracle_probabilities[action]
            cumulative_rewards[name].append(reward_totals[name])
            cumulative_regrets[name].append(regret_totals[name])
        progress_bar.progress(
            (round_index + 1) / n_rounds,
            text=f"Round {round_index + 1} of {n_rounds}",
        )

    return {
        "cumulative_rewards": cumulative_rewards,
        "cumulative_regrets": cumulative_regrets,
        "total_rewards": reward_totals,
        "avg_regrets": {name: regret_totals[name] / n_rounds for name in POLICY_NAMES},
        "decisions": decisions,
        "linucb": policies["LinUCB"],
    }


def make_reward_chart(result, selected_policy, n_rounds, inspected_round=None, mode="reward"):
    figure = go.Figure()
    rounds = np.arange(1, n_rounds + 1)
    metric_key = "cumulative_rewards" if mode == "reward" else "cumulative_regrets"
    ylabel = "Cumulative likes" if mode == "reward" else "Cumulative regret gap"
    for name in POLICY_NAMES:
        chosen = name == selected_policy
        color = POLICY_COLORS[name]
        figure.add_trace(
            go.Scatter(
                x=rounds,
                y=result[metric_key][name],
                name=name,
                mode="lines",
                line={"color": color, "width": 3.5 if chosen else 1.5},
                opacity=1.0 if chosen else 0.28,
                fill="tozeroy" if chosen else None,
                fillcolor=("rgba(245,158,11,0.10)" if chosen else None),
                hovertemplate=f"{name}<br>Round %{{x}}<br>{ylabel}: %{{y:.3f}}<extra></extra>",
            )
        )
    figure.update_layout(
        height=390,
        margin={"l": 8, "r": 12, "t": 18, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT, "family": "Inter, sans-serif"},
        xaxis={"title": "Round", "gridcolor": "rgba(148,163,184,.10)", "zeroline": False},
        yaxis={"title": ylabel, "gridcolor": "rgba(148,163,184,.10)", "zeroline": False},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0, "bgcolor": "rgba(0,0,0,0)"},
        hoverlabel={"bgcolor": "#111827", "font_color": TEXT, "bordercolor": "rgba(255,255,255,.12)"},
    )
    if inspected_round is not None:
        figure.add_vline(x=inspected_round, line_width=1, line_dash="dash", line_color="rgba(226,232,240,.45)")
    return figure


def make_genre_chart(context, weights=None, title="Genre profile"):
    values = np.asarray(context if weights is None else weights, dtype=float)
    rows = pd.DataFrame({"Genre": GENRE_NAMES, "Weight": values})
    if weights is None:
        rows = rows[rows.Weight > 0].sort_values("Weight", ascending=True).tail(12)
        color = CYAN
        hover = "%{y}<br>Profile weight: %{x:.3f}<extra></extra>"
        figure = go.Figure(
            go.Bar(x=rows.Weight, y=rows.Genre, orientation="h", marker_color=color, hovertemplate=hover)
        )
        figure.update_xaxes(title="Share of liked genres", range=[0, max(0.15, float(rows.Weight.max()) * 1.15)])
    else:
        rows = rows.sort_values("Weight", ascending=True)
        colors = [CYAN if value >= 0 else "#E48672" for value in rows.Weight]
        figure = go.Figure(
            go.Bar(
                x=rows.Weight,
                y=rows.Genre,
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}<br>Estimated coefficient: %{x:.3f}<extra></extra>",
            )
        )
        figure.update_xaxes(title="Estimated reward weight", zeroline=True, zerolinecolor="#64737C")
    figure.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15}},
        height=390 if weights is not None else 330,
        margin={"l": 8, "r": 18, "t": 45, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT, "family": "Inter, sans-serif"},
        yaxis={"title": "", "gridcolor": "rgba(0,0,0,0)"},
        hoverlabel={"bgcolor": "#111827", "font_color": TEXT},
    )
    return figure


def metric_card(label, value, help_text=None):
    st.markdown(
        f'<div class="kpi-card" title="{escape(str(help_text or ""), quote=True)}"><div class="kpi-label">{escape(str(label))}</div><div class="kpi-value">{escape(str(value))}</div><div class="kpi-sub">{escape(str(help_text or "Movie Bandit Lab telemetry"))}</div></div>',
        unsafe_allow_html=True,
    )


def release_year(title):
    match = re.search(r"\((\d{4})\)", str(title))
    return match.group(1) if match else "Year n/a"


def set_inspector_round(state_key, round_number):
    """Set the inspector widget value before Streamlit reruns the script."""
    st.session_state[state_key] = round_number


def render_movie_rows(rows):
    cards = []
    for rank, row in enumerate(rows, start=1):
        title = str(row["Movie"])
        probability = float(row["Oracle p(like) raw"])
        cards.append(
            '<div class="movie-row-card"><div class="movie-top"><div><span class="rank">#%02d</span><span class="movie-title">%s</span> <span class="year-pill">%s</span></div><span class="source-pill">%s</span></div><div class="prob-line"><span>ORACLE P(LIKE)</span><span>%.1f%%</span></div><div class="prob-track"><div class="prob-fill" style="width:%.1f%%"></div></div><div class="movie-meta"><span class="mono-tag">Genre match · %s</span><span class="mono-tag">Popularity · %s</span></div></div>'
            % (rank, escape(title), escape(release_year(title)), "Bollywood Demo" if int(row["Movie ID"]) < 0 else "MovieLens 100K", probability * 100, probability * 100, row["Genre match"], row["Popularity"])
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


def oracle_breakdown(user_id, movie_id):
    profile = user_contexts[user_id] if user_id in user_contexts else demo_user_contexts[user_id]
    vector = movie_genre_vectors[movie_id]
    overlap = float(profile @ vector)
    popularity = float(movie_popularity_score.get(movie_id, 0.0))
    genre_logit = 4.0 * overlap
    popularity_logit = popularity
    baseline = -1.5
    logit = genre_logit + popularity_logit + baseline
    probability = 1.0 / (1.0 + np.exp(-logit))
    return overlap, genre_logit, popularity_logit, baseline, probability


def render_explanation(user_id, result, selected_profile):
    linucb_policy = result["linucb"]
    context = MovieBanditEnvironment._context_for_user(user_id)
    means, bonuses, scores = linucb_scores(linucb_policy, context)
    arm = int(np.argmax(scores))
    movie_id = int(candidate_movies[arm])
    inverse = np.linalg.inv(linucb_policy.A[arm])
    weights = inverse @ linucb_policy.b[arm]
    overlap, genre_term, popularity_term, intercept, oracle_probability = oracle_breakdown(user_id, movie_id)

    st.markdown("#### LinUCB's current choice")
    st.caption(
        f"For {selected_profile}, LinUCB's highest upper-confidence score is **{movie_titles[movie_id]}**."
    )
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.plotly_chart(
            make_genre_chart(weights, weights=weights, title="Learned genre weights for this movie"),
            config={"displayModeBar": False},
        )
        count = sum(
            decision["movie_id"] == movie_id
            for decision in result["decisions"]["LinUCB"]
        )
        st.caption(
            f"These coefficients come from {count} observed rewards for this movie. The model uses them with this user's genre profile."
        )
    with right:
        metric_card("LinUCB estimated reward", f"{means[arm]:.3f}", "Learned mean reward for this movie and user context.")
        metric_card("Exploration bonus", f"+{bonuses[arm]:.3f}", "LinUCB's uncertainty bonus for this arm and user context.")
        mean_width = float(np.clip(abs(means[arm]) / (abs(means[arm]) + bonuses[arm] + 1e-9), 0.05, 0.95) * 100)
        bonus_width = 100 - mean_width
        math_card = f'''<div class="bento-card"><div class="decision-label">Math &amp; telemetry breakdown</div><div class="equation" style="margin-top:12px"><span class="eq-block">Learned Mean θᵀx<br><b>{means[arm]:.3f}</b></span><b style="color:#64748B">+</b><span class="eq-block">α · Uncertainty Bonus<br><b>{bonuses[arm]:.3f}</b></span><b style="color:#64748B">=</b><span class="eq-block">Total UCB Score<br><b>{scores[arm]:.3f}</b></span></div><div class="split-track" style="height:7px;margin:12px 0 18px"><div class="mean-fill" style="width:{mean_width:.1f}%"></div><div class="bonus-fill" style="width:{bonus_width:.1f}%"></div></div><div class="decision-label">Environment oracle logit</div><div class="equation" style="margin-top:10px"><span class="eq-block">σ(4.0 × {overlap:.3f}</span><span class="eq-block">+ 1.0 × {movie_popularity_score.get(movie_id, 0.0):.3f}</span><span class="eq-block">− 1.5)</span><span style="color:#64748B">=</span><span class="eq-block" style="color:#6EE7B7">p(like) {oracle_probability:.1%}</span></div><div class="kpi-sub" style="margin-top:12px">Genre contribution {genre_term:+.3f} · popularity contribution {popularity_term:+.3f} · intercept {intercept:+.3f}. Popularity shapes the oracle, not LinUCB features.</div></div>'''
        st.markdown(math_card, unsafe_allow_html=True)
        st.caption(
            "The genre and popularity lines explain the oracle's sampled reward probability. LinUCB's action score itself is its learned estimate plus the exploration bonus; popularity is not a direct LinUCB feature."
        )


def render_how_it_works():
    st.markdown("### How it works")
    with st.expander("The bandit problem", expanded=False):
        st.write(
            "Each round, a policy recommends one movie from the fixed slate. The environment samples a like or no-like reward from a frozen probability. The policy sees that reward and updates its state."
        )
    with st.expander("Epsilon-Greedy and LinUCB", expanded=False):
        st.write(
            "Epsilon-Greedy mostly chooses the movie with its best average reward, but explores a random movie 10% of the time. LinUCB keeps a separate linear estimate for each movie and adds an uncertainty bonus, so it can try choices that may fit a less-seen user's context."
        )
    with st.expander("Why context matters", expanded=False):
        st.write(
            "A user's training profile is a normalized count of genres in movies they rated 4 or 5. LinUCB receives that profile; Epsilon-Greedy deliberately ignores it. This lets us compare learning one average for everyone with learning patterns that vary by taste."
        )
    with st.expander("What the rewards mean", expanded=False):
        st.write(
            "Rewards are simulated Bernoulli samples from a fixed, hand-built response rule. MovieLens titles use training-only genre and like-count signals. Bollywood titles are hand-added demo items with illustrative genre flags and popularity scores; they have no MovieLens ratings."
        )


ratings, display_movies = load_movie_data(str(DATA_DIR))
titles = display_movies.set_index("item_id").title.to_dict()
titles.update(movie_titles)

with st.sidebar:
    st.markdown("## Movie Bandit Lab")
    st.caption("A small, inspectable recommendation experiment")
    st.markdown('<div class="decision-label">01 / VIEWER CONTEXT</div>', unsafe_allow_html=True)
    selected_profile = st.selectbox("Simulated viewer", list(PROFILE_USERS))
    selected_user_id = PROFILE_USERS[selected_profile]
    viewer_context = MovieBanditEnvironment._context_for_user(selected_user_id)
    viewer_genres = np.argsort(viewer_context)[::-1]
    viewer_top_genres = [GENRE_NAMES[i] for i in viewer_genres if viewer_context[i] > 0][:3]
    viewer_type = f"Synthetic Demo Profile · Bollywood Buff" if selected_user_id == BOLLYWOOD_USER_ID else f"Real MovieLens User · ID #{selected_user_id}"
    genre_pills = "".join('<span class="mono-tag">%s</span>' % escape(g) for g in viewer_top_genres)
    st.markdown(f'<div class="bento-card" style="padding:12px;margin:8px 0 18px"><div class="decision-label">Viewer context summary</div><div style="color:#E2E8F0;font-size:.78rem;margin:8px 0">{escape(viewer_type)}</div><div class="movie-meta">{genre_pills}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="decision-label">02 / FOCUS POLICY</div>', unsafe_allow_html=True)
    selected_policy = option_menu(
        "Policy",
        POLICY_NAMES,
        icons=["shuffle", "bar-chart-fill", "arrow-repeat", "cpu"],
        menu_icon="film",
        default_index=3,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": CYAN, "font-size": "15px"},
            "nav-link": {"color": TEXT, "font-size": "14px", "text-align": "left", "margin": "2px", "--hover-color": "#263540"},
            "nav-link-selected": {"background-color": "#34414A", "color": SAFFRON, "font-weight": "700"},
        },
    )
    st.markdown('<div class="decision-label" style="margin-top:18px">03 / HORIZON &amp; EXECUTION</div>', unsafe_allow_html=True)
    number_of_rounds = st.slider("Rounds per policy", 10, 500, 120, step=10)
    run_clicked = st.button("Run the experiment", type="primary", use_container_width=True)
    st.markdown("<hr class='hero-rule'>", unsafe_allow_html=True)
    st.caption("Same user · same frozen oracle · independent policy learning")


st.markdown(f'''<div class="hero-status">● &nbsp; CONTEXTUAL BANDIT TELEMETRY · 42-ARM SLATE · MOVIELENS 100K + BOLLYWOOD SYNTHETIC</div><div class="hero-title">Learning what a viewer likes.</div><p class="hero-copy">A research interface for the exploration–exploitation tradeoff. Compare four recommendation policies, follow stochastic feedback, and inspect exactly how context and uncertainty shape each choice.</p><div class="walkthrough"><span class="tiny-pill">01 &nbsp; Choose viewer · {escape(selected_profile)}</span><span class="tiny-pill">02 &nbsp; Focus policy · {escape(selected_policy)}</span><span class="tiny-pill">03 &nbsp; Horizon · {number_of_rounds} rounds</span><span class="tiny-pill">2,200 total configurations</span></div>''', unsafe_allow_html=True)
st.markdown("<hr class='hero-rule'>", unsafe_allow_html=True)

recommendation_env = MovieBanditEnvironment(random_state=0)
recommendation_probabilities = recommendation_env.predict_probabilities(selected_user_id)
top_indices = np.argsort(recommendation_probabilities)[::-1][:5]
top_recommendations = pd.DataFrame(
    {
        "Movie": [titles[int(candidate_movies[index])] for index in top_indices],
        "Oracle p(like)": [recommendation_probabilities[index] for index in top_indices],
        "Genre match": [
            float(
                MovieBanditEnvironment._context_for_user(selected_user_id)
                @ movie_genre_vectors[int(candidate_movies[index])]
            )
            for index in top_indices
        ],
        "Popularity": [movie_popularity_score[int(candidate_movies[index])] for index in top_indices],
        "Movie ID": [int(candidate_movies[index]) for index in top_indices],
        "Oracle p(like) raw": [recommendation_probabilities[index] for index in top_indices],
    }
)
top_recommendations["Oracle p(like)"] = top_recommendations["Oracle p(like)"].map(lambda value: f"{value:.1%}")
top_recommendations["Genre match"] = top_recommendations["Genre match"].map(lambda value: f"{value:.3f}")
top_recommendations["Popularity"] = top_recommendations["Popularity"].map(lambda value: f"{value:.2f}")

expected_key = (selected_profile, number_of_rounds)
if run_clicked:
    progress = st.progress(0.0, text="Preparing the experiment")
    try:
        result = run_demo(selected_user_id, number_of_rounds, seed=17, progress_bar=progress)
        st.session_state["demo_result"] = {"key": expected_key, "result": result}
    except Exception as error:
        st.error(f"The experiment could not finish: {error}")
    finally:
        progress.empty()

saved = st.session_state.get("demo_result")
has_matching_result = bool(saved and saved["key"] == expected_key)

liked_genre_count = int(np.count_nonzero(MovieBanditEnvironment._context_for_user(selected_user_id)))
inspected_round = st.session_state.get(f"inspector-{selected_profile}-{selected_policy}-{number_of_rounds}", number_of_rounds)
selected_total = saved["result"]["total_rewards"][selected_policy] if has_matching_result else None
selected_regret = saved["result"]["avg_regrets"][selected_policy] if has_matching_result else None

metric_columns = st.columns(4, gap="medium")
with metric_columns[0]:
    metric_card("Movie slate", f"{len(candidate_movies)}", f"{len(BOLLYWOOD_ITEM_IDS)} hand-added demo movies.")
with metric_columns[1]:
    metric_card("Liked genres", f"{liked_genre_count}", "Genres with non-zero weight in this user's profile.")
with metric_columns[2]:
    metric_card("Likes · selected policy", str(selected_total) if selected_total is not None else "—", "Observed simulated likes from the selected policy.")
with metric_columns[3]:
    metric_card("Average regret", f"{selected_regret:.3f}" if selected_regret is not None else "—", "Mean oracle probability gap per round; lower is better.")

st.markdown("<hr class='hero-rule'>", unsafe_allow_html=True)
chart_column, profile_column = st.columns([1.55, 1], gap="large")
with chart_column:
    st.subheader("Learning trajectory")
    chart_mode = st.radio("Trajectory", ["Cumulative Likes (Reward)", "Cumulative Regret Gap"], horizontal=True, label_visibility="collapsed", key="trajectory-mode")
    trajectory_mode = "reward" if chart_mode.startswith("Cumulative Likes") else "regret"
    st.markdown(
        f"<p class='section-note'>Selected policy: <b style='color:{POLICY_COLORS[selected_policy]}'>{escape(selected_policy)}</b>. Fainter lines are the comparison policies.</p>",
        unsafe_allow_html=True,
    )
    if has_matching_result:
        st.plotly_chart(
            make_reward_chart(saved["result"], selected_policy, number_of_rounds, inspected_round, trajectory_mode),
            config={"displayModeBar": False, "scrollZoom": False},
        )
    else:
        st.info("Run the experiment to draw the cumulative reward curves.")

with profile_column:
    st.subheader("This viewer's taste")
    st.caption(f"{selected_profile} · training profile")
    st.plotly_chart(
        make_genre_chart(MovieBanditEnvironment._context_for_user(selected_user_id)),
        config={"displayModeBar": False},
    )

st.markdown("<hr class='hero-rule'>", unsafe_allow_html=True)
recommendation_column, explanation_column = st.columns([1, 1.5], gap="large")
with recommendation_column:
    st.subheader("Most likely to land")
    st.caption("The fixed oracle's top five for this viewer, before the run.")
    render_movie_rows(top_recommendations.to_dict("records"))
with explanation_column:
    st.subheader("What LinUCB learned")
    if has_matching_result:
        render_explanation(selected_user_id, saved["result"], selected_profile)
    else:
        st.info("Run the experiment to inspect LinUCB's learned genre weights and current choice.")

st.markdown("<hr class='hero-rule'>", unsafe_allow_html=True)
st.subheader("Round-by-round inspector")
if has_matching_result:
    inspector_key = f"inspector-{selected_profile}-{selected_policy}-{number_of_rounds}"
    early_col, late_col, _ = st.columns([1, 1, 5])
    with early_col:
        st.button(
            "↤ Early exploration",
            key=f"jump-early-{selected_profile}-{selected_policy}-{number_of_rounds}",
            on_click=set_inspector_round,
            args=(inspector_key, 1),
        )
    with late_col:
        st.button(
            "Late convergence ↦",
            key=f"jump-late-{selected_profile}-{selected_policy}-{number_of_rounds}",
            on_click=set_inspector_round,
            args=(inspector_key, number_of_rounds),
        )
    inspected_round = st.slider(
        "Choose a round",
        min_value=1,
        max_value=number_of_rounds,
        value=number_of_rounds,
        key=inspector_key,
        help="Inspect the selected policy's movie choice, exploration mode, oracle probability, and sampled reward for this round.",
    )
    decision = saved["result"]["decisions"][selected_policy][inspected_round - 1]
    movie_id = int(decision["movie_id"])
    genre_tags = [GENRE_NAMES[i] for i, value in enumerate(movie_genre_vectors[movie_id]) if value > 0]
    explore = decision["behavior"] == "Exploration"
    reward_badge = '<span class="badge-liked">LIKED (+1)</span>' if decision["reward"] else '<span class="badge-disliked">NO LIKE (0)</span>'
    mode_badge = '<span class="badge-explore">EXPLORATION</span>' if explore else '<span class="badge-exploit">EXPLOITATION</span>'
    decision_extra = f'<div class="equation" style="margin-top:12px"><span class="eq-block">Estimate {decision["linucb_estimate"]:.3f}</span><b>+</b><span class="eq-block">Bonus {decision["ucb_bonus"]:.3f}</span><b>=</b><span class="eq-block">Score {decision["ucb_score"]:.3f}</span></div>' if selected_policy == "LinUCB" else ''
    genre_html = "".join('<span class="mono-tag">%s</span>' % escape(g) for g in genre_tags)
    decision_html = f'''<div class="decision-card"><div><div class="decision-label">Arm #{movie_id if movie_id >= 0 else abs(movie_id)} · Round {inspected_round}</div><div class="decision-title">{escape(decision["movie"])}</div><span class="year-pill">{escape(release_year(decision["movie"]))}</span> <span class="source-pill">{"Bollywood Demo" if movie_id < 0 else "MovieLens 100K"}</span><div class="genre-tags">{genre_html}</div></div><div><div class="decision-label">Decision telemetry</div><div style="margin:12px 0">{mode_badge}</div><div class="prob-line"><span>ORACLE P(LIKE)</span><span>{decision["oracle_p_like"]:.1%}</span></div><div class="prob-track" style="height:9px"><div class="prob-fill" style="width:{decision["oracle_p_like"]*100:.1f}%"></div></div>{decision_extra}</div><div><div class="decision-label">Bernoulli outcome</div><div style="margin:15px 0">{reward_badge}</div><div class="kpi-sub">One stochastic sample from the frozen oracle. A single outcome can differ from the displayed probability.</div></div></div>'''
    st.markdown(decision_html, unsafe_allow_html=True)
else:
    st.caption("After a run, move the slider to review the selected policy's movie, oracle probability, reward, and exploration choice.")

if has_matching_result:
    st.markdown("<hr class='hero-rule'>", unsafe_allow_html=True)
    st.subheader("Policy scorecard")
    best_policy = max(POLICY_NAMES, key=lambda name: saved["result"]["total_rewards"][name])
    leaderboard = []
    for name in POLICY_NAMES:
        decisions = saved["result"]["decisions"][name]
        explore_pct = 100 * sum(item["behavior"] == "Exploration" for item in decisions) / len(decisions)
        regret = saved["result"]["avg_regrets"][name]
        regret_width = min(100, regret / max(saved["result"]["avg_regrets"].values() or [1]) * 100) if max(saved["result"]["avg_regrets"].values()) else 0
        best = '<div class="best-badge">✦ BEST PERFORMER</div>' if name == best_policy else ''
        leaderboard.append(f'<div class="leader-card"><div class="leader-policy"><span style="color:{POLICY_COLORS[name]}">●</span> &nbsp;{escape(name)}</div>{best}<div class="leader-value">{saved["result"]["total_rewards"][name]} <span style="color:#64748B;font-size:.65rem">LIKES</span></div><div class="prob-line"><span>AVG REGRET</span><span>{regret:.3f}</span></div><div class="regret-track"><div style="width:{regret_width:.1f}%;height:100%;background:{POLICY_COLORS[name]};border-radius:99px"></div></div><div class="prob-line" style="margin-top:12px"><span>EXPLORATION</span><span>{explore_pct:.1f}%</span></div></div>')
    st.markdown('<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px">'+''.join(leaderboard)+'</div>', unsafe_allow_html=True)

render_how_it_works()
with st.expander("Explore the 42-Movie Action Slate", expanded=False):
    st.caption("Each title's fixed oracle probability is computed for the currently selected viewer.")
    slate_cards = []
    for index, movie_id in enumerate(candidate_movies):
        movie_id = int(movie_id)
        title = movie_titles[movie_id]
        genres = [GENRE_NAMES[i] for i, value in enumerate(movie_genre_vectors[movie_id]) if value > 0]
        probability = float(recommendation_probabilities[index])
        genre_html = "".join('<span class="mono-tag">%s</span>' % escape(g) for g in genres)
        slate_cards.append(f'<div class="movie-row-card"><div class="movie-top"><span class="movie-title">{escape(title)} <span class="year-pill">{escape(release_year(title))}</span></span><span class="source-pill">{"Bollywood Demo" if movie_id < 0 else "MovieLens 100K"}</span></div><div class="prob-line"><span>ORACLE P(LIKE)</span><span>{probability:.1%}</span></div><div class="prob-track"><div class="prob-fill" style="width:{probability*100:.1f}%"></div></div><div class="movie-meta"><span class="mono-tag">Popularity · {movie_popularity_score[movie_id]:.2f}</span>{genre_html}</div></div>')
    st.markdown('<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px">'+''.join(slate_cards)+'</div>', unsafe_allow_html=True)
st.markdown(
    "<div class='footer-note'>Rewards are simulated from a fixed oracle; Bollywood titles are hand-added demo items with illustrative metadata and no MovieLens ratings.</div>",
    unsafe_allow_html=True,
)
