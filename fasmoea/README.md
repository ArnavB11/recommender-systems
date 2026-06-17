# FAS-MOEA Implementation

Implementation of the FAS-MOEA framework from:

> Khaitan, S., & Shrivastava, R. (2026). Developing Fairness, Accuracy, and
> Serendipity Objective Functions for Recommendation System and Establishing
> Trade-off through Multi-Objective Evolutionary Optimization. *Information
> Processing and Management*, 63, 104604.

## Architecture

FAS-MOEA uses NSGA-II to jointly optimize three objectives:

1. **Accuracy**: weighted blend of long-tail and non-long-tail item predictions
2. **Fairness**: per-user genre distribution alignment with the global catalog
3. **Serendipity**: novelty times context multiplier for unexpected relevant items

## Files

- `fasmoea_model.py`: three objective function implementations
- `fasmoea_ga.py`: pymoo NSGA-II problem, FAS crossover, and targeted novelty mutation
- `fasmoea_runner.py`: orchestration function used by `main.py`

## Usage

From the project root:

```bash
python main.py
```

This runs both MSRS and FAS-MOEA end-to-end and writes all outputs under
`results/`.
