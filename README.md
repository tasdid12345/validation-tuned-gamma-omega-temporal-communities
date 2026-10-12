# Validation-selected resolution and coupling for multilayer community detection on a messaging network

Community detection on temporal networks depends on two knobs: a resolution parameter that sets community granularity and a coupling parameter that sets how strongly communities persist across time. The paper "Generalized L-Modularity for Community Detection Beyond Simple Temporal Networks" (2026) motivates treating these parameters explicitly when interactions are timestamped and directed. This experiment asks a narrower question with a standard tool.

## Question
Can choosing the resolution γ and temporal coupling ω on an earlier validation window give communities that better explain later interactions than the default γ=1, ω=1 setting?

## Dataset
UCI/SNAP CollegeMsg temporal messaging network, downloaded over HTTP from snap.stanford.edu. It is a public dataset and not the paper's own data (`same_as_paper: false`). Messages are split by time into training, validation and test windows.

## Baseline vs improved model
- **Baseline:** multilayer (Mucha-style) Louvain over time slices with default γ and ω.
- **Improved:** the same detector, with γ and ω chosen from a small grid on the validation window, scoring intra-community fraction minus a partition-instability penalty.

Both models use the same data loading, splits, evaluation function and seeds. The test window is scored once.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json`. Charts are saved to `figures/`.

## Results (latest run)

On the public CollegeMsg network, a standard multilayer Louvain at default γ=1, ω=1 reached 34.5% test intra-community interaction fraction, versus 40.2% with γ=0.5, ω=0.25 chosen on validation; a modest 5.7-point gain on one dataset over 3 seeds.

| Metric | Value |
|---|---|
| baseline_score | 0.3451 |
| improved_score | 0.4023 |
| improvement | 0.05713 |
| baseline_score_std | 0.002797 |
| improved_score_std | 0 |
| baseline_n_communities | 271.3 |
| improved_n_communities | 19.67 |
| baseline_stability_ari | 0.9603 |
| improved_stability_ari | 0.999 |
| selected_gamma | 0.5 |
| selected_omega | 0.25 |

Run on 2026-10-12. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Louvain is applied to the whole supra-graph with a single resolution parameter, an approximation of Mucha's layer-wise null model; this is not the LAGO algorithm of the motivating paper. Interactions are treated as undirected and unweighted by direction. Communities are fixed from the training window; each user takes its label from the latest training slice; users unseen in training are singletons. The validation window is a time-ordered part of the non-test data, never the test set. Single dataset, small grid, 3 seeds; differences may be small relative to seed variation.

## Reference

This project is motivated by the following paper:

> V Brabant, A Bonifati, R Cazabet (2026). *Generalized L-Modularity for Community Detection Beyond Simple Temporal Networks*. arXiv preprint arXiv:2605.24450. https://arxiv.org/abs/2605.24450

Paper author: Prof. Angela Bonifati, Université Claude Bernard Lyon 1 (angela.bonifati@univ-lyon1.fr)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
