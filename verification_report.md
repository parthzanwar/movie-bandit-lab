# Evaluation verification report

Generated: 2026-10-21

## Run configuration

- Candidate slate: 42 movies (30 MovieLens candidates plus 12 hand-added Bollywood demo items).
- Seeds: 0, 1, 2, 3, 4.
- Rounds: 2,000 per policy per seed (10,000 rounds per policy total).
- Results are recomputed by `run_evaluation.py` on every run. Oracle probabilities are cached only in memory within that run. No serialized evaluation-result cache files were present. The fresh run overwrote `results_reward.png` and `results_regret.png`.

## Exact evaluated slate

| Index | Movie | Item ID | Source |
|---:|---|---:|---|
| 0 | Empire Strikes Back, The (1980) | 172 | MovieLens 100K |
| 1 | L.A. Confidential (1997) | 302 | MovieLens 100K |
| 2 | Aladdin (1992) | 95 | MovieLens 100K |
| 3 | Scream (1996) | 288 | MovieLens 100K |
| 4 | E.T. the Extra-Terrestrial (1982) | 423 | MovieLens 100K |
| 5 | Dances with Wolves (1990) | 97 | MovieLens 100K |
| 6 | Hoop Dreams (1994) | 48 | MovieLens 100K |
| 7 | Star Wars (1977) | 50 | MovieLens 100K |
| 8 | Fargo (1996) | 100 | MovieLens 100K |
| 9 | Contact (1997) | 258 | MovieLens 100K |
| 10 | Return of the Jedi (1983) | 181 | MovieLens 100K |
| 11 | English Patient, The (1996) | 286 | MovieLens 100K |
| 12 | Liar Liar (1997) | 294 | MovieLens 100K |
| 13 | Air Force One (1997) | 300 | MovieLens 100K |
| 14 | Toy Story (1995) | 1 | MovieLens 100K |
| 15 | Independence Day (ID4) (1996) | 121 | MovieLens 100K |
| 16 | Raiders of the Lost Ark (1981) | 174 | MovieLens 100K |
| 17 | Godfather, The (1972) | 127 | MovieLens 100K |
| 18 | Twelve Monkeys (1995) | 7 | MovieLens 100K |
| 19 | Pulp Fiction (1994) | 56 | MovieLens 100K |
| 20 | Silence of the Lambs, The (1991) | 98 | MovieLens 100K |
| 21 | Jerry Maguire (1996) | 237 | MovieLens 100K |
| 22 | Rock, The (1996) | 117 | MovieLens 100K |
| 23 | Star Trek: First Contact (1996) | 222 | MovieLens 100K |
| 24 | Back to the Future (1985) | 204 | MovieLens 100K |
| 25 | Princess Bride, The (1987) | 173 | MovieLens 100K |
| 26 | Forrest Gump (1994) | 69 | MovieLens 100K |
| 27 | Fugitive, The (1993) | 79 | MovieLens 100K |
| 28 | Mission: Impossible (1996) | 405 | MovieLens 100K |
| 29 | Indiana Jones and the Last Crusade (1989) | 210 | MovieLens 100K |
| 30 | 3 Idiots (2009) | -1 | Hand-added demo item |
| 31 | Dangal (2016) | -2 | Hand-added demo item |
| 32 | Sholay (1975) | -3 | Hand-added demo item |
| 33 | Dil Chahta Hai (2001) | -4 | Hand-added demo item |
| 34 | Gully Boy (2019) | -5 | Hand-added demo item |
| 35 | Queen (2013) | -6 | Hand-added demo item |
| 36 | Andhadhun (2018) | -7 | Hand-added demo item |
| 37 | Lagaan (2001) | -8 | Hand-added demo item |
| 38 | PK (2014) | -9 | Hand-added demo item |
| 39 | Zindagi Na Milegi Dobara (2011) | -10 | Hand-added demo item |
| 40 | Chak De! India (2007) | -11 | Hand-added demo item |
| 41 | Barfi! (2012) | -12 | Hand-added demo item |

The Bollywood titles were included in the evaluated slate, not just the app.
They have no MovieLens ratings: their genre flags and popularity scores are
hand-assigned demo values.

## Oracle and reward source

The current oracle is a fixed, hand-specified logistic scoring rule, not a
logistic regression fitted to Bollywood or MovieLens rating labels:

```text
p(like) = sigmoid(4.0 * dot(user_genre_profile, movie_genre_vector)
                  + 1.0 * movie_popularity_score - 1.5)
```

- For MovieLens candidates, user genre profiles come from training ratings of
  4 or 5, movie genre vectors come from `u.item`, and popularity is the
  training like count min-max normalized over the original 30 candidates.
- For Bollywood demo candidates, genre vectors were manually mapped into the
  same 19 MovieLens genres and popularity values were manually assigned on the
  `[0, 1]` scale. The same formula generates their probabilities.
- Each selected movie's reward is sampled as a Bernoulli draw using that
  probability. The Bollywood buff is a synthetic demo user and is not sampled
  by evaluation, which draws users from MovieLens training users only.

## Fresh run and LinUCB audit

The evaluation did run again. The script has no disk result cache; it creates
new policies and environments for every invocation. No cached evaluation
results existed to delete. The plot files are outputs, and the fresh run
overwrote both.

LinUCB selected **zero Bollywood movies** across all five seeds (0 of 10,000
rounds). That explains why its reward mean and standard deviation stayed
exactly the same as before: its sampled actions stayed in the MovieLens part
of the slate. The identical LinUCB numbers were not evidence of a stale result.

| Policy | Mean total reward | Std across seeds |
|---|---:|---:|
| Random | 1092.0 | 16.8 |
| Popularity | 1517.8 | 19.5 |
| Epsilon-Greedy | 1496.8 | 23.7 |
| LinUCB | 1586.8 | 19.4 |

Bollywood actions across the five seeds were: Random 2,890; Popularity 10,000;
Epsilon-Greedy 3,917; LinUCB 0.

Fresh plots: `results_reward.png` and `results_regret.png`.
