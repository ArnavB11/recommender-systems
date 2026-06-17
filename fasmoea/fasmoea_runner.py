"""
FAS-MOEA runner used by main.py.

The runner receives data, R_hat, and eval users from the unified entry point so
shared objects are not loaded or recomputed twice.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fasmoea.fasmoea_ga import run_fasmoea_for_user
from fasmoea.fasmoea_model import FASMOEAObjectives
from ncf_inference import get_pure_signal

N_CANDIDATE_POOL = int(os.environ.get("FASMOEA_CANDIDATE_POOL", "200"))
N_MAX_REC = 30
POP_SIZE = int(os.environ.get("FASMOEA_POP_SIZE", "50"))
N_GENERATIONS = int(os.environ.get("FASMOEA_GENERATIONS", "100"))
SEED = 42


def run_fasmoea_pipeline(data, R_hat, eval_users):
    """
    Run FAS-MOEA for all evaluation users at N=30.

    Returns:
        fasmoea_recs: user_idx -> recommendation list of item_idx values
        fasmoea_pareto: user_idx -> Pareto objective dictionaries
        objectives: fitted FASMOEAObjectives instance
    """
    objectives = FASMOEAObjectives(data, R_hat).fit()
    fasmoea_recs = {}
    fasmoea_pareto = {}

    print("\n=== [FAS-MOEA] Running NSGA-II optimization per user ===")
    for pos, user_idx in enumerate(eval_users, start=1):
        print(f"  FAS-MOEA user {user_idx} ({pos}/{len(eval_users)})", end="\r")
        pure_signal = get_pure_signal(R_hat[user_idx], data.user_history.get(user_idx, set()))
        candidate_pool = pure_signal["item_indices"][:N_CANDIDATE_POOL]

        if len(candidate_pool) < 10:
            fasmoea_recs[int(user_idx)] = [int(item_idx) for item_idx in candidate_pool]
            fasmoea_pareto[int(user_idx)] = []
            continue

        best_list, pareto_metrics = run_fasmoea_for_user(
            user_idx=int(user_idx),
            candidate_pool=candidate_pool,
            objectives=objectives,
            R_hat_user=R_hat[user_idx],
            N=min(N_MAX_REC, len(candidate_pool)),
            pop_size=POP_SIZE,
            n_generations=N_GENERATIONS,
            seed=SEED,
        )
        fasmoea_recs[int(user_idx)] = [int(item_idx) for item_idx in best_list]
        fasmoea_pareto[int(user_idx)] = pareto_metrics

    print(f"\n[FAS-MOEA] Complete for {len(eval_users)} users.")
    return fasmoea_recs, fasmoea_pareto, objectives
