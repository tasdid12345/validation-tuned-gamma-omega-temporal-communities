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

## Reference

This project is motivated by the following paper:

> V Brabant, A Bonifati, R Cazabet (2026). *Generalized L-Modularity for Community Detection Beyond Simple Temporal Networks*. arXiv preprint arXiv:2605.24450. https://arxiv.org/abs/2605.24450

Paper author: Prof. Angela Bonifati, Université Claude Bernard Lyon 1 (angela.bonifati@univ-lyon1.fr)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
