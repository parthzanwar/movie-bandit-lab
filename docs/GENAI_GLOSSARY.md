# Movie Bandit Lab: plain-language glossary

## Reward and like probability

For one selected movie, the simulator draws a binary reward. Reward `1` means
the simulated viewer liked that movie; reward `0` means they did not. The draw
is Bernoulli with the movie's oracle `p(like)`. For example, if `p(like) = 0.8`,
the simulator is more likely to return a like, but a no-like can still happen.

The reward is feedback for that one choice. It is not a label saying the policy
made the best possible choice, and it is not a real-world accuracy score.

## Oracle probability and regret gap

The oracle probability is the simulator's fixed expected like chance for a
viewer/movie pair. Regret for one round is:

```text
best oracle p(like) for this viewer - chosen movie's oracle p(like)
```

Average regret is the mean of that gap across rounds. Lower regret means the
policy chose movies whose expected like chances were closer to the simulator's
best available chance. It is different from the number of observed no-likes:
regret compares probabilities, while reward records the random outcome.

## Accuracy and real viewers

This project does not observe a live viewer or know what a real person wants.
Its oracle is a hand-specified simulation rule, so regret measures performance
against that simulator only. A high number of likes in one run is not proof of
real-world recommendation accuracy. Robust comparisons require repeated seeds,
and real accuracy claims require an appropriate real-user evaluation.

## Exploration and exploitation

Exploration means trying an option to learn more about it. Exploitation means
choosing an option that current evidence suggests is good. Epsilon-Greedy
explores randomly some of the time; LinUCB adds an uncertainty bonus to its
learned expected reward to favor less-tested options when that may be useful.

## LinUCB estimate and uncertainty bonus

For a movie arm `a` and the current 19-genre context `x`, LinUCB estimates a
mean reward with `theta_a = inverse(A_a) * b_a`, then computes
`mean_a = theta_a dot x`. Its uncertainty bonus is
`alpha * sqrt(x dot inverse(A_a) dot x)`, and its action score is mean plus
bonus. Each movie has its own `A_a` and `b_a`; only the selected movie's model
is updated from that round's simulated reward. The bonus represents model
uncertainty for this context. It is not a count of how many real times the
viewer rated that movie, because the app does not collect live viewer ratings.

## Retrieval-Augmented Generation

RAG first searches project notes for relevant passages, then gives those
passages to a language model as evidence for its answer. In this project,
TF-IDF word/phrase vectors and cosine similarity rank the passages. The tutor
returns source IDs so a student can inspect where an explanation came from.
