#!/usr/bin/env python
"""
run_comprehensive_comparison.py

Unified evaluation script: runs 9 baselines + Pure NCF + FAS-MOEA + MSRS (MSRS II/IV excluded).
Supports configurable user counts and sets exactly 100 generations (or customizable) for all GAs.
Generates all 14 individual top-N curves, K=10 grouped bar charts, and radar plots.
"""

import os
import sys
import time
import argparse
import json
import math
import pickle
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure testing directory is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data_loader import MovieLensData, load_ratings
from ncf_inference import get_pure_signal, load_trained_model
from dataset_linker import DatasetLinker
from dopm import DOPMRecommender
from serendipity import SerendipityModel
from fairness import FairnessObjective
from psnr_filter import filter_by_psnr_sweetspot
from weighted_roulette import generate_weighted_roulette_lists
from genetic_algorithm import run_nsga2_optimization, run_nsga2_optimization_msrs_iv

# Import Baselines
from baselines.cf_utils import (
    get_genre_matrix, get_test_relevant_items, get_train_user_history,
    get_candidate_pool, popularity_fallback_topn
)
from baselines.pmoea import PMOEA
from baselines.chmaor import CHMAOR
from baselines.maora_imf import MaORAIMF
from baselines.morem import MOREM
from baselines.cmoea_penn import CMOEAPeNN
from baselines.ae_mc_nsga2 import AEMCNSGA2
from baselines.moea_miae import MOEAMIAE
from baselines.crossgcl import CrossGCL
from baselines.dcrlrec import DCRLRec

# Import FAS-MOEA components
from fasmoea.fasmoea_ga import run_fasmoea_for_user
from fasmoea.fasmoea_model import FASMOEAObjectives

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
RAW_DIR = os.path.join(RESULTS_DIR, "comprehensive_raw")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

K_VALUES = [5, 10, 15, 20, 25, 30]
N_MAX_REC = 30
EXTENDED_N = 30
SEED = 42

# Color configurations — MSRS (Proposed) is ONLY bold bright red (#FF0000), MOREM is changed to pink (#E377C2)
METHOD_DISPLAY_NAMES = {
    "pmoea": "PMOEA",
    "chmaor": "CHMAOR",
    "maora_imf": "MaORA-IMF",
    "morem": "MOREM",
    "cmoea_penn": "CMOEA-PeNN",
    "ae_mc_nsga2": "AE-MC+NSGA-II",
    "moea_miae": "MOEA-MIAE",
    "crossgcl": "CrossGCL",
    "dcrlrec": "DCRLRec",
    "pure_ncf": "Pure NCF",
    "fasmoea": "FAS-MOEA",
    "msrs": "MSRS (Proposed)",
    "msrs_iv": "MSRS IV",
}

METHOD_STYLE = {
    "pmoea":         {"color": "#555555", "marker": ".", "linestyle": ":",  "label": "PMOEA",          "lw": 1.5, "zorder": 2},
    "chmaor":        {"color": "#ff7f0e", "marker": "s", "linestyle": "-",  "label": "CHMAOR",         "lw": 1.5, "zorder": 2},
    "maora_imf":     {"color": "#9c27b0", "marker": "^", "linestyle": "-",  "label": "MaORA-IMF",      "lw": 1.5, "zorder": 2},
    "morem":         {"color": "#e377c2", "marker": "D", "linestyle": "-",  "label": "MOREM",          "lw": 1.5, "zorder": 2},  # Non-red pink
    "cmoea_penn":    {"color": "#9467bd", "marker": "P", "linestyle": "-",  "label": "CMOEA-PeNN",     "lw": 1.5, "zorder": 2},
    "ae_mc_nsga2":   {"color": "#8c564b", "marker": "X", "linestyle": "-",  "label": "AE-MC+NSGA-II",  "lw": 1.5, "zorder": 2},
    "moea_miae":     {"color": "#bcbd22", "marker": "*", "linestyle": "-",  "label": "MOEA-MIAE",      "lw": 1.5, "zorder": 2},
    "crossgcl":      {"color": "#7f7f7f", "marker": "v", "linestyle": "-",  "label": "CrossGCL",       "lw": 1.5, "zorder": 2},
    "dcrlrec":       {"color": "#008080", "marker": "h", "linestyle": "-",  "label": "DCRLRec",        "lw": 1.5, "zorder": 2},
    "pure_ncf":      {"color": "#17becf", "marker": "<", "linestyle": "--", "label": "Pure NCF",       "lw": 1.5, "zorder": 2},
    "fasmoea":       {"color": "#2196F3", "marker": "o", "linestyle": "-",  "label": "FAS-MOEA",       "lw": 2.0, "zorder": 3},
    "msrs":          {"color": "#FF0000", "marker": "*", "linestyle": "-",  "label": "MSRS (Proposed)", "lw": 3.5, "zorder": 10}, # Boldest red
    "msrs_iv":       {"color": "#4CAF50", "marker": "D", "linestyle": "-.", "label": "MSRS IV",         "lw": 2.5, "zorder": 5},
}

METHOD_ORDER = [
    "pmoea", "chmaor", "maora_imf", "morem", "cmoea_penn",
    "ae_mc_nsga2", "moea_miae", "crossgcl", "dcrlrec",
    "pure_ncf", "fasmoea", "msrs", "msrs_iv"
]

# =====================================================================
# Metric Implementations matching testing/main.py
# =====================================================================

def precision_at_k(rec_list, ground_truth, k):
    rec_k = rec_list[:k]
    if not rec_k:
        return 0.0
    return len(set(rec_k) & ground_truth) / k

def recall_at_k(rec_list, ground_truth, k):
    rec_k = rec_list[:k]
    if not ground_truth:
        return 0.0
    return len(set(rec_k) & ground_truth) / len(ground_truth)

def dcg_at_k(rec_list, ground_truth, k):
    rec_k = rec_list[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in ground_truth:
            dcg += 1.0 / math.log2(i + 2)
    return dcg

def ndcg_at_k(rec_list, ground_truth, k):
    actual_dcg = dcg_at_k(rec_list, ground_truth, k)
    ideal_hits = min(len(ground_truth), k)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def ap_at_k(rec_list, ground_truth, k):
    rec_k = rec_list[:k]
    hits = 0
    ap = 0.0
    for i, item in enumerate(rec_k):
        if item in ground_truth:
            hits += 1
            ap += hits / (i + 1)
    return ap / min(len(ground_truth), k) if ground_truth else 0.0

def f1_at_k(rec_list, ground_truth, k):
    precision = precision_at_k(rec_list, ground_truth, k)
    recall = recall_at_k(rec_list, ground_truth, k)
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def ild_at_k(rec_list, data, k):
    rec_k = rec_list[:k]
    genres = set()
    for item_idx in rec_k:
        item_id = data.idx2item.get(item_idx)
        if item_id and item_id in data.items_df.index:
            primary_genre = data.items_df.loc[item_id, "primary_genre"]
            if isinstance(primary_genre, str):
                genres.add(primary_genre)
    return len(genres) / k if k > 0 else 0.0

def novelty_at_k(rec_list, data, k):
    rec_k = rec_list[:k]
    if not rec_k:
        return 0.0
    max_pop = data.pop_max or 1
    scores = []
    for item_idx in rec_k:
        pop = data.item_popularity.get(item_idx, 0)
        novelty = 1.0 - (math.log(pop + 1) / math.log(max_pop + 1))
        scores.append(novelty)
    return float(np.mean(scores))

def serendipity_at_k(rec_list, user_idx, objectives, k):
    return objectives.compute_serendipity(user_idx, rec_list[:k])

def long_tail_coverage_at_k(rec_list, long_tail_items, k):
    rec_k = rec_list[:k]
    if not rec_k:
        return 0.0
    return len([item_idx for item_idx in rec_k if item_idx in long_tail_items]) / k

def genre_coverage_at_k(all_recs_k, data):
    all_genres_dataset = set()
    for item_id in data.items_df.index:
        genres_str = data.items_df.loc[item_id, "genres"]
        if isinstance(genres_str, str):
            for genre in genres_str.split("|"):
                if genre and genre != "unknown":
                    all_genres_dataset.add(genre)

    rec_genres = set()
    for rec_k in all_recs_k:
        for item_idx in rec_k:
            item_id = data.idx2item.get(item_idx)
            if item_id and item_id in data.items_df.index:
                genres_str = data.items_df.loc[item_id, "genres"]
                if isinstance(genres_str, str):
                    for genre in genres_str.split("|"):
                        if genre and genre != "unknown":
                            rec_genres.add(genre)
    return len(rec_genres) / len(all_genres_dataset) if all_genres_dataset else 0.0

def catalog_coverage_at_k(all_recs_k, data):
    unique_items = set()
    for rec_k in all_recs_k:
        unique_items.update(rec_k)
    return len(unique_items) / data.n_items if data.n_items > 0 else 0.0

def dopm_at_k(rec_list, data, R_hat, user_idx, dopm_model, k):
    rec_k = rec_list[:k]
    if not rec_k:
        return 0.0
    scores = [
        dopm_model.calculate_dopm(user_idx, item_idx, float(R_hat[user_idx, item_idx]))
        for item_idx in rec_k
    ]
    scores = [score for score in scores if score is not None]
    return float(np.mean(scores)) if scores else 0.0

def genre_entropy_at_k(rec_list, data, k):
    rec_k = rec_list[:k]
    genre_counts = {}
    total = 0
    for item_idx in rec_k:
        item_id = data.idx2item.get(item_idx)
        if item_id and item_id in data.items_df.index:
            primary_genre = data.items_df.loc[item_id, "primary_genre"]
            if isinstance(primary_genre, str):
                genre_counts[primary_genre] = genre_counts.get(primary_genre, 0) + 1
                total += 1
    if total == 0:
        return 0.0
    entropy = -sum(
        (count / total) * math.log2(count / total)
        for count in genre_counts.values()
        if count > 0
    )
    return float(entropy)

def fas_fairness_at_k(rec_list, user_idx, objectives, k):
    return objectives.compute_fairness(user_idx, rec_list[:k])

def producer_fairness_at_k(rec_list, data, R_hat, user_idx, dopm_model, objectives, fairness_model, k):
    rec_k = rec_list[:k]
    dopm_dict = {
        item_idx: dopm_model.calculate_dopm(user_idx, item_idx, float(R_hat[user_idx, item_idx])) or 0.0
        for item_idx in rec_k
    }
    ser_dict = {
        item_idx: objectives.compute_serendipity(user_idx, [item_idx])
        for item_idx in rec_k
    }
    return fairness_model.compute_fairness(rec_k, dopm_dict, ser_dict)

# =====================================================================
# Comprehensive Evaluator
# =====================================================================

def evaluate_model_comprehensive(recommendations, data, R_hat, ground_truth, eval_users, dopm_model, objectives, fairness_model):
    model_results = {}
    for k in K_VALUES:
        metric_values = defaultdict(list)
        all_recs_k = []

        for user_idx in eval_users:
            rec_list = [int(item_idx) for item_idx in recommendations.get(int(user_idx), [])]
            gt = ground_truth.get(user_idx, set())
            rec_k = rec_list[:k]
            all_recs_k.append(rec_k)

            metric_values["Precision@K"].append(precision_at_k(rec_list, gt, k))
            metric_values["Recall@K"].append(recall_at_k(rec_list, gt, k))
            metric_values["NDCG@K"].append(ndcg_at_k(rec_list, gt, k))
            metric_values["MAP@K"].append(ap_at_k(rec_list, gt, k))
            metric_values["F1@K"].append(f1_at_k(rec_list, gt, k))
            metric_values["ILD@K"].append(ild_at_k(rec_list, data, k))
            metric_values["Serendipity@K"].append(serendipity_at_k(rec_list, user_idx, objectives, k))
            metric_values["Novelty@K"].append(novelty_at_k(rec_list, data, k))
            metric_values["DOPM@K"].append(dopm_at_k(rec_list, data, R_hat, user_idx, dopm_model, k))
            metric_values["LongTailCoverage@K"].append(
                long_tail_coverage_at_k(rec_list, objectives.long_tail_items, k)
            )
            metric_values["FAS_Fairness@K"].append(fas_fairness_at_k(rec_list, user_idx, objectives, k))
            metric_values["GenreEntropy@K"].append(genre_entropy_at_k(rec_list, data, k))
            metric_values["ProducerFairness@K"].append(
                producer_fairness_at_k(rec_list, data, R_hat, user_idx, dopm_model, objectives, fairness_model, k)
            )

        row = {
            metric: float(np.mean(values)) if values else 0.0
            for metric, values in metric_values.items()
        }
        row["GenreCoverage@K"] = genre_coverage_at_k(all_recs_k, data)
        row["CatalogCoverage@K"] = catalog_coverage_at_k(all_recs_k, data)
        model_results[k] = row

    return model_results

# =====================================================================
# Individual Plots, Radar and Grouped Bar Generation
# =====================================================================

def plot_metric_topn(df_all, metric_name, ylabel, title, filename, figs_dir):
    groups = {
        "group1": ["pmoea", "chmaor", "maora_imf"],
        "group2": ["morem", "cmoea_penn", "ae_mc_nsga2"],
        "group3": ["moea_miae", "crossgcl", "dcrlrec"]
    }
    benchmarks = ["pure_ncf", "fasmoea", "msrs", "msrs_iv"]
    
    base_name, ext = os.path.splitext(filename)
    
    for group_name, group_baselines in groups.items():
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        models_to_plot = group_baselines + benchmarks
        
        for model in METHOD_ORDER:
            if model not in models_to_plot:
                continue
            if model not in df_all["Model"].values:
                continue
            sub = df_all[df_all["Model"] == model].sort_values("K")
            style = METHOD_STYLE[model]
            ax.plot(
                sub["K"],
                sub[metric_name],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=style["lw"],
                markersize=8 if model in ["msrs", "msrs_iv"] else 6,
                label=style["label"],
                zorder=style["zorder"]
            )
        ax.set_xlabel("Top-N Recommendations", fontsize=11, fontweight="semibold")
        ax.set_ylabel(ylabel, fontsize=11, fontweight="semibold")
        group_display_label = group_name.replace("group", "Group ")
        ax.set_title(f"{title} ({group_display_label})", fontsize=12, fontweight="bold")
        ax.set_xticks(K_VALUES)
        ax.legend(fontsize=9, frameon=True, facecolor="white", edgecolor="#e0e0e0")
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        
        group_filename = f"{base_name}_{group_name}{ext}"
        plt.savefig(os.path.join(figs_dir, group_filename), dpi=300, bbox_inches="tight")
        plt.close()

def plot_radar_summary(df_all, figs_dir):
    metrics = [
        "Precision@K",
        "NDCG@K",
        "ILD@K",
        "Novelty@K",
        "Serendipity@K",
        "LongTailCoverage@K",
        "FAS_Fairness@K",
        "GenreEntropy@K",
        "ProducerFairness@K",
        "DOPM@K",
    ]
    labels = [
        "Precision",
        "NDCG",
        "ILD",
        "Novelty",
        "Serendipity",
        "Long-tail",
        "FAS Fairness",
        "Entropy",
        "Producer Fairness",
        "DOPM",
    ]
    df_k10 = df_all[df_all["K"] == 10].copy()

    normalized = {}
    for metric in metrics:
        values = df_k10[metric].astype(float)
        min_val = float(values.min())
        max_val = float(values.max())
        if max_val == min_val:
            normalized[metric] = {model: 1.0 for model in df_k10["Model"]}
        else:
            normalized[metric] = {
                row["Model"]: (float(row[metric]) - min_val) / (max_val - min_val)
                for _, row in df_k10.iterrows()
            }

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})

    for model in METHOD_ORDER:
        if model not in df_k10["Model"].values:
            continue
        style = METHOD_STYLE[model]
        values = [normalized[metric].get(model, 0.0) for metric in metrics]
        values += values[:1]
        ax.plot(angles, values, color=style["color"], linewidth=style["lw"], linestyle=style["linestyle"], label=style["label"], marker=style["marker"], markersize=8 if model in ["msrs", "msrs_iv"] else 4, zorder=style["zorder"])
        ax.fill(angles, values, color=style["color"], alpha=0.03)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("K=10 Normalized Metric Radar", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "fig_radar_summary.png"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_grouped_bar_k10(df_all, figs_dir):
    groups = [
        ("Consumer Accuracy", ["Precision@K", "Recall@K", "NDCG@K", "F1@K"]),
        ("Consumer Beyond-Acc", ["ILD@K", "Novelty@K", "Serendipity@K"]),
        ("Provider", ["DOPM@K", "LongTailCoverage@K"]),
        ("System Fairness", ["FAS_Fairness@K", "GenreEntropy@K", "ProducerFairness@K"]),
    ]
    df_k10 = df_all[df_all["K"] == 10].set_index("Model")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5))

    active_models = [m for m in METHOD_ORDER if m in df_k10.index]
    n_models = len(active_models)
    
    # Calculate bar positions dynamically to avoid overlap
    width = 0.8 / n_models

    for ax, (title, metrics) in zip(axes, groups):
        x = np.arange(len(metrics))
        
        for idx, model in enumerate(active_models):
            style = METHOD_STYLE[model]
            offset = (idx - n_models / 2 + 0.5) * width
            vals = [float(df_k10.loc[model, metric]) for metric in metrics]
            ax.bar(
                x + offset, 
                vals, 
                width, 
                label=style["label"], 
                color=style["color"], 
                edgecolor="white", 
                linewidth=0.5
            )
            
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([metric.replace("@K", "") for metric in metrics], rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.35)

    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("K=10 Stakeholder Metric Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "fig_grouped_bar_k10.png"), dpi=300, bbox_inches="tight")
    plt.close()

# =====================================================================
# Helper Pipeline Runners for Comprehensive Evaluation
# =====================================================================

def ensure_recommendation_length(rec_list, fallback_pool, n=30):
    """Deduplicate and fill short recommendation lists from a fallback pool."""
    seen = set()
    cleaned = []
    for item_idx in rec_list:
        item_idx = int(item_idx)
        if item_idx not in seen:
            cleaned.append(item_idx)
            seen.add(item_idx)
        if len(cleaned) >= n:
            return cleaned

    for item_idx in fallback_pool:
        item_idx = int(item_idx)
        if item_idx not in seen:
            cleaned.append(item_idx)
            seen.add(item_idx)
        if len(cleaned) >= n:
            break
    return cleaned


def generate_full_r_hat(model, n_users, n_items, batch_size=65536):
    """Generate the shared NCF score matrix once, without retraining."""
    total_pairs = n_users * n_items
    scores = np.empty(total_pairs, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, total_pairs, batch_size):
            end = min(start + batch_size, total_pairs)
            flat_idx = torch.arange(start, end, dtype=torch.long)
            users = flat_idx // n_items
            items = flat_idx % n_items
            batch_scores = model(users, items).detach().cpu().numpy()
            scores[start:end] = batch_scores.astype(np.float32)

    return scores.reshape(n_users, n_items)


def run_baseline_comp(cls, data, user_indices, seed=SEED, R_hat=None, **kwargs):
    """Fits and runs a baseline recommender model."""
    if R_hat is not None:
        model = cls(data, R_hat=R_hat, **kwargs)
    else:
        model = cls(data, **kwargs)
    
    model.fit(seed=seed)
    
    seed_recs = model.recommend_all_users(user_indices, n=10)
    seed_ext = {}
    for u in user_indices:
        seed_ext[u] = model.recommend_extended(u, n=30)
        
    return seed_recs, seed_ext, 0.0


def run_pure_ncf(data, R_hat, user_indices, train_history, n=30):
    """Generates Pure NCF recommendations."""
    ncf_recs = {}
    for u in user_indices:
        history = train_history.get(u, set())
        pure_signal = get_pure_signal(R_hat[u], history)
        ncf_recs[u] = pure_signal["item_indices"][:n]
    return ncf_recs


def run_fasmoea_pipeline_comp(data, R_hat, eval_users, objectives, seed=SEED, n_gen=100):
    """Runs FAS-MOEA optimization for evaluation users."""
    fasmoea_recs = {}
    for user_idx in eval_users:
        pure_signal = get_pure_signal(R_hat[user_idx], data.user_history.get(user_idx, set()))
        candidate_pool = pure_signal["item_indices"][:200]
        
        if len(candidate_pool) == 0:
            fasmoea_recs[user_idx] = []
            continue
            
        best_list, _ = run_fasmoea_for_user(
            user_idx=user_idx,
            candidate_pool=candidate_pool,
            objectives=objectives,
            R_hat_user=R_hat[user_idx],
            N=min(30, len(candidate_pool)),
            pop_size=50,
            n_generations=n_gen,
            seed=seed
        )
        fasmoea_recs[user_idx] = best_list
    return fasmoea_recs


def run_msrs_pipeline_comp(data, R_hat, model, dopm_system, serendipity_system, fairness_model, user_indices, train_history, seed, n_gen):
    """Runs standard PSNR-filtered MSRS NSGA-II pipeline."""
    msrs_recs = {}
    for user_idx in user_indices:
        pure_signal = get_pure_signal(R_hat[user_idx], train_history.get(user_idx, set()))
        candidate_pool = pure_signal["item_indices"][:200]
        candidate_scores = np.asarray(pure_signal["scores"][:200], dtype=np.float64)

        if len(candidate_pool) == 0:
            msrs_recs[int(user_idx)] = []
            continue

        rec_size = min(30, len(candidate_pool))
        if len(candidate_pool) <= rec_size:
            msrs_recs[int(user_idx)] = [int(item_idx) for item_idx in candidate_pool]
            continue

        movie_scores_dict = {
            int(item_idx): float(score)
            for item_idx, score in zip(candidate_pool, candidate_scores)
        }

        roulette_lists = generate_weighted_roulette_lists(
            pure_signal_scores=candidate_scores.reshape(1, -1),
            user_watch_matrix=np.zeros((1, len(candidate_pool)), dtype=np.int8),
            users=[int(user_idx)],
            movies=candidate_pool,
            N=rec_size,
            variations=1000,
            pool_size=min(50, len(candidate_pool)),
        )

        original_ncf_list = candidate_pool[:rec_size]
        accepted_variants, _, _ = filter_by_psnr_sweetspot(
            original_list=original_ncf_list,
            variant_lists=roulette_lists.get(int(user_idx), []),
            movie_scores_dict=movie_scores_dict,
            lower_percentile=10,
            upper_percentile=90,
            max_val=1.0,
        )

        initial_variants = [
            ensure_recommendation_length(variant["list"], candidate_pool, rec_size)
            for variant in accepted_variants
            if variant.get("list")
        ]
        if not initial_variants:
            initial_variants = [original_ncf_list]

        best_list, _ = run_nsga2_optimization(
            user_idx=int(user_idx),
            candidate_pool=candidate_pool,
            ncf_scores=R_hat[user_idx],
            dopm_recommender=dopm_system,
            serendipity_model=serendipity_system,
            fairness_model=fairness_model,
            user_history=list(train_history.get(user_idx, set())),
            initial_population_variants=initial_variants,
            N=rec_size,
            pop_size=50,
            n_generations=n_gen,
        )
        msrs_recs[int(user_idx)] = best_list
    return msrs_recs


def run_msrs_iv_pipeline_comp(data, R_hat, dopm_system, serendipity_system, fairness_model, user_indices, train_history, seed, n_gen):
    """Runs MSRS IV pipeline using novelty seeding and relevance-constrained targeted mutation."""
    msrs_iv_recs = {}
    item_popularity = data.item_popularity
    max_popularity = data.pop_max or 1
    
    for user_idx in user_indices:
        pure_signal = get_pure_signal(R_hat[user_idx], train_history.get(user_idx, set()))
        candidate_pool = pure_signal["item_indices"][:200]
        
        if len(candidate_pool) == 0:
            msrs_iv_recs[int(user_idx)] = []
            continue
            
        rec_size = min(30, len(candidate_pool))
        if len(candidate_pool) <= rec_size:
            msrs_iv_recs[int(user_idx)] = [int(item_idx) for item_idx in candidate_pool]
            continue
            
        best_list, _ = run_nsga2_optimization_msrs_iv(
            user_idx=int(user_idx),
            candidate_pool=candidate_pool,
            ncf_scores=R_hat[user_idx],
            dopm_recommender=dopm_system,
            serendipity_model=serendipity_system,
            fairness_model=fairness_model,
            user_history=list(train_history.get(user_idx, set())),
            item_popularity=item_popularity,
            max_popularity=max_popularity,
            N=rec_size,
            pop_size=50,
            n_generations=n_gen,
            seed=seed
        )
        msrs_iv_recs[int(user_idx)] = best_list
    return msrs_iv_recs


# =====================================================================
# Strata User Sampling
# =====================================================================

def sample_users(data, n_users=943, seed=SEED):
    train_history = get_train_user_history(data)
    ratings_df = load_ratings()
    ratings_df["user_idx"] = ratings_df["user_id"].map(data.user2idx)
    ratings_df["item_idx"] = ratings_df["item_id"].map(data.item2idx)
    
    from sklearn.model_selection import train_test_split
    _, holdout_df = train_test_split(ratings_df, test_size=0.2, random_state=seed)
    ground_truth = holdout_df[holdout_df["rating"] >= 4].groupby("user_idx")["item_idx"].apply(set).to_dict()
    
    eligible_users = [u for u in range(data.n_users) if ground_truth.get(u)]
    if n_users >= len(eligible_users):
        return sorted(eligible_users), ground_truth
        
    rng = np.random.RandomState(seed)
    counts = np.array([len(train_history.get(u, set())) for u in eligible_users])
    quartiles = np.percentile(counts[counts > 0], [25, 50, 75])
    bins = np.digitize(counts, quartiles)

    sampled = []
    per_quartile = n_users // 4
    for q in range(4):
        q_indices = np.where(bins == q)[0]
        if len(q_indices) > 0:
            n_sample = min(per_quartile, len(q_indices))
            sampled_indices = rng.choice(q_indices, size=n_sample, replace=False)
            sampled.extend([eligible_users[idx] for idx in sampled_indices])

    remaining = n_users - len(sampled)
    if remaining > 0:
        all_remaining = set(eligible_users) - set(sampled)
        if all_remaining:
            sampled.extend(rng.choice(list(all_remaining), size=min(remaining, len(all_remaining)), replace=False).tolist())

    return sorted(sampled[:n_users]), ground_truth

# =====================================================================
# Main Pipeline Function
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation of baseline models vs. MSRS and FAS-MOEA")
    parser.add_argument("--n_users", type=int, default=943, help="Number of users to evaluate (default: all)")
    parser.add_argument("--n_seeds", type=int, default=1, help="Number of seeds")
    parser.add_argument("--quick", action="store_true", help="Smoke test mode (fewer epochs/generations)")
    parser.add_argument("--n_gen", type=int, default=100, help="Number of generations for evolutionary solvers")
    args = parser.parse_args()

    n_gen = 10 if args.quick else args.n_gen
    n_epochs_crossgcl = 5 if args.quick else 30
    n_epochs_dcrlrec = 5 if args.quick else 20
    reinforce_steps = 2 if args.quick else 5

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 80)
    print(f"COMPREHENSIVE ALL-USER COMPARISON: n_users={args.n_users}, n_gen={n_gen}")
    print("=" * 80)

    # 1. Load Data
    data = MovieLensData()
    model = load_trained_model(data)
    
    print("\nGenerating NeuMF R_hat Matrix...")
    R_hat = generate_full_r_hat(model, data.n_users, data.n_items)
    
    user_indices, ground_truth = sample_users(data, n_users=args.n_users)
    print(f"Evaluating {len(user_indices)} eligible users out of {data.n_users}")

    # Set data.user_history to train-only history
    train_history = get_train_user_history(data)
    data.user_history = train_history

    # Initialize objectives
    print("Initializing DOPM, Serendipity, and Fairness Objectives...")
    dopm_model = DOPMRecommender()
    movie_genres = {item_idx: data.items_df.loc[data.idx2item[item_idx], "genres"].split("|") for item_idx in range(data.n_items)}
    dopm_model.fit(train_history, movie_genres)

    linker = DatasetLinker()
    serendipity_model = SerendipityModel(data=data, linker=linker)
    
    fairness_model = FairnessObjective(data)
    fairness_model.fit()

    # Initialize FAS-MOEA objectives
    fasmoea_objectives = FASMOEAObjectives(data, R_hat).fit()

    results = {}

    baseline_configs = {
        "pmoea": {
            "cls": PMOEA,
            "kwargs": {"pop_size": 100, "n_gen": n_gen},
        },
        "chmaor": {
            "cls": CHMAOR,
            "kwargs": {"n_gen": n_gen},
        },
        "maora_imf": {
            "cls": MaORAIMF,
            "kwargs": {"n_gen": n_gen},
        },
        "morem": {
            "cls": MOREM,
            "kwargs": {"pop_size": 50, "n_gen": n_gen},
            "needs_R_hat": True,
        },
        "cmoea_penn": {
            "cls": CMOEAPeNN,
            "kwargs": {"pop_size": 80, "n_gen": n_gen},
            "needs_R_hat": True,
        },
        "ae_mc_nsga2": {
            "cls": AEMCNSGA2,
            "kwargs": {"pop_size": 80, "n_gen": n_gen, "ae_epochs": 10 if args.quick else 30},
        },
        "moea_miae": {
            "cls": MOEAMIAE,
            "kwargs": {"sub_pop_size": 40, "n_gen": n_gen},
            "needs_R_hat": True,
        },
        "crossgcl": {
            "cls": CrossGCL,
            "kwargs": {"n_epochs": n_epochs_crossgcl},
        },
        "dcrlrec": {
            "cls": DCRLRec,
            "kwargs": {"n_epochs": n_epochs_dcrlrec, "reinforce_steps": reinforce_steps},
        },
    }

    # =========================================================================
    # Run loop
    # =========================================================================

    # 1. Standard GAs and Direct models
    for name, config in baseline_configs.items():
        print(f"\n[RUNNING] Baseline: {name}")
        cls = config["cls"]
        kwargs = config["kwargs"]
        needs_r_hat = config.get("needs_R_hat", False)

        recs = {}
        for seed in range(args.n_seeds):
            print(f"  Seed {seed}...")
            #fit once
            if needs_r_hat:
                seed_recs, seed_ext, _ = run_baseline_comp(cls, data, user_indices, seed=seed, R_hat=R_hat, **kwargs)
            else:
                seed_recs, seed_ext, _ = run_baseline_comp(cls, data, user_indices, seed=seed, **kwargs)
            
            # Combine or use the first seed for comparison curves
            for u in user_indices:
                recs[u] = seed_ext[u]

        results[name] = evaluate_model_comprehensive(
            recs, data, R_hat, ground_truth, user_indices, dopm_model, fasmoea_objectives, fairness_model
        )

    # 2. Pure NCF Baseline
    print("\n[RUNNING] Pure NCF Baseline")
    ncf_recs = run_pure_ncf(data, R_hat, user_indices, train_history, n=EXTENDED_N)
    results["pure_ncf"] = evaluate_model_comprehensive(
        ncf_recs, data, R_hat, ground_truth, user_indices, dopm_model, fasmoea_objectives, fairness_model
    )

    # 3. FAS-MOEA Proposed
    print("\n[RUNNING] FAS-MOEA Proposed")
    fas_recs_raw = run_fasmoea_pipeline_comp(data, R_hat, user_indices, objectives=fasmoea_objectives, seed=SEED, n_gen=n_gen)
    fas_recs = {}
    for u_idx in user_indices:
        base_list = fas_recs_raw.get(u_idx, [])
        if len(base_list) < EXTENDED_N:
            full_candidates = get_candidate_pool(u_idx, R_hat[u_idx], train_history, pool_size=EXTENDED_N + len(base_list))
            extension = [i for i in full_candidates if i not in set(base_list)]
            fas_recs[u_idx] = base_list + extension[:EXTENDED_N - len(base_list)]
        else:
            fas_recs[u_idx] = base_list[:EXTENDED_N]

    results["fasmoea"] = evaluate_model_comprehensive(
        fas_recs, data, R_hat, ground_truth, user_indices, dopm_model, fasmoea_objectives, fairness_model
    )

    # 4. MSRS Proposed
    print("\n[RUNNING] MSRS Proposed")
    msrs_recs_raw = run_msrs_pipeline_comp(
        data=data, R_hat=R_hat, model=model, dopm_system=dopm_model, serendipity_system=serendipity_model,
        fairness_model=fairness_model, user_indices=user_indices, train_history=train_history, seed=SEED, n_gen=n_gen
    )
    msrs_recs = {}
    for u_idx in user_indices:
        base_list = msrs_recs_raw.get(u_idx, [])
        if len(base_list) < EXTENDED_N:
            full_candidates = get_candidate_pool(u_idx, R_hat[u_idx], train_history, pool_size=EXTENDED_N + len(base_list))
            extension = [i for i in full_candidates if i not in set(base_list)]
            msrs_recs[u_idx] = base_list + extension[:EXTENDED_N - len(base_list)]
        else:
            msrs_recs[u_idx] = base_list[:EXTENDED_N]

    results["msrs"] = evaluate_model_comprehensive(
        msrs_recs, data, R_hat, ground_truth, user_indices, dopm_model, fasmoea_objectives, fairness_model
    )

    # 5. MSRS IV
    print("\n[RUNNING] MSRS IV")
    msrs_iv_recs_raw = run_msrs_iv_pipeline_comp(
        data=data, R_hat=R_hat, dopm_system=dopm_model, serendipity_system=serendipity_model,
        fairness_model=fairness_model, user_indices=user_indices, train_history=train_history, seed=SEED, n_gen=n_gen
    )
    msrs_iv_recs = {}
    for u_idx in user_indices:
        base_list = msrs_iv_recs_raw.get(u_idx, [])
        if len(base_list) < EXTENDED_N:
            full_candidates = get_candidate_pool(u_idx, R_hat[u_idx], train_history, pool_size=EXTENDED_N + len(base_list))
            extension = [i for i in full_candidates if i not in set(base_list)]
            msrs_iv_recs[u_idx] = base_list + extension[:EXTENDED_N - len(base_list)]
        else:
            msrs_iv_recs[u_idx] = base_list[:EXTENDED_N]

    results["msrs_iv"] = evaluate_model_comprehensive(
        msrs_iv_recs, data, R_hat, ground_truth, user_indices, dopm_model, fasmoea_objectives, fairness_model
    )

    # Save Pickle Data
    with open(os.path.join(RAW_DIR, "results_by_model.pkl"), "wb") as f:
        pickle.dump(results, f)

    # =========================================================================
    # Write Tables & Draw Visualizations
    # =========================================================================
    print("\nWriting Summary Tables...")
    rows = []
    for model_name in METHOD_ORDER:
        if model_name not in results:
            continue
        for k in K_VALUES:
            row = {"Model": model_name, "K": k}
            row.update(results[model_name][k])
            rows.append(row)
    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(TABLES_DIR, "comprehensive_summary_metrics.csv"), index=False)

    print("\nPlotting Figure Curves...")
    figure_specs = [
        ("Precision@K", "Precision", "Precision@K vs Top-N Recommendations", "fig_precision_topn.png"),
        ("Recall@K", "Recall", "Recall@K vs Top-N Recommendations", "fig_recall_topn.png"),
        ("NDCG@K", "NDCG", "NDCG@K vs Top-N Recommendations", "fig_ndcg_topn.png"),
        ("MAP@K", "MAP", "MAP@K vs Top-N Recommendations", "fig_map_topn.png"),
        ("F1@K", "F1-Score", "F1@K vs Top-N Recommendations", "fig_f1_topn.png"),
        ("ILD@K", "Intra-List Diversity", "ILD@K (Genre Diversity) vs Top-N", "fig_ild_topn.png"),
        ("Serendipity@K", "Serendipity Score", "Serendipity@K vs Top-N Recommendations", "fig_serendipity_topn.png"),
        ("Novelty@K", "Novelty Score", "Novelty@K vs Top-N Recommendations", "fig_novelty_topn.png"),
        ("DOPM@K", "DOPM Score", "DOPM@K vs Top-N Recommendations", "fig_dopm_topn.png"),
        ("LongTailCoverage@K", "Long-Tail Coverage", "Long-Tail Coverage@K vs Top-N", "fig_longtail_coverage_topn.png"),
        ("CatalogCoverage@K", "Catalog Coverage", "Catalog Coverage@K vs Top-N", "fig_catalog_coverage_topn.png"),
        ("FAS_Fairness@K", "FAS Fairness Score", "FAS-MOEA Fairness@K vs Top-N", "fig_fas_fairness_topn.png"),
        ("GenreEntropy@K", "Genre Entropy (bits)", "Genre Entropy@K vs Top-N Recommendations", "fig_genre_entropy_topn.png"),
        ("ProducerFairness@K", "Producer Fairness (HDB)", "Producer (HDB) Fairness@K vs Top-N", "fig_producer_fairness_topn.png"),
    ]

    for metric, ylabel, title, filename in figure_specs:
        plot_metric_topn(df_all, metric, ylabel, title, filename, FIGURES_DIR)

    plot_radar_summary(df_all, FIGURES_DIR)
    plot_grouped_bar_k10(df_all, FIGURES_DIR)

    print("\n" + "=" * 80)
    print("ALL EVALUATIONS SUCCESSFUL! RUN COMPLETE.")
    print(f"Summary Table saved: {TABLES_DIR}/comprehensive_summary_metrics.csv")
    print(f"All Figures saved under: {FIGURES_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
