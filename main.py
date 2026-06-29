import json
import math
import os
import shutil
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from data_loader import MovieLensData, load_ratings
from dataset_linker import DatasetLinker
from dopm import DOPMRecommender
from fairness import FairnessObjective
from fasmoea.fasmoea_runner import run_fasmoea_pipeline
from genetic_algorithm import run_nsga2_optimization
from ncf_inference import get_pure_signal, load_trained_model
from psnr_filter import filter_by_psnr_sweetspot
from serendipity import SerendipityModel
from weighted_roulette import generate_weighted_roulette_lists

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

SEED = 42
K_VALUES = [5, 10, 15, 20, 25, 30]
N_MAX_REC = 30
N_CANDIDATE_POOL = int(os.environ.get("MSRS_CANDIDATE_POOL", "200"))

MSRS_VARIATIONS = int(os.environ.get("MSRS_VARIATIONS", "1000"))
MSRS_ROULETTE_POOL_SIZE = int(os.environ.get("MSRS_ROULETTE_POOL_SIZE", "50"))
MSRS_POP_SIZE = int(os.environ.get("MSRS_POP_SIZE", "50"))
MSRS_GENERATIONS = int(os.environ.get("MSRS_GENERATIONS", "40"))


def prepare_results_dirs():
    """Clear and recreate the required tables and figures subfolders, preserving predictions."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    for path in [TABLES_DIR, FIGURES_DIR]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def save_json(path, payload):
    def convert(value):
        if isinstance(value, dict):
            return {str(key): convert(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        return value

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(convert(payload), handle, indent=2)


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


def build_ground_truth_and_train_history(data):
    """
    Build an 80/20 holdout and switch data.user_history to train-only history.

    Held-out items rated >= 4 are relevant. Using train-only history keeps those
    held-out relevant items eligible for recommendation during evaluation.
    """
    ratings_df = load_ratings()
    ratings_df["user_idx"] = ratings_df["user_id"].map(data.user2idx)
    ratings_df["item_idx"] = ratings_df["item_id"].map(data.item2idx)

    train_df, holdout_df = train_test_split(
        ratings_df,
        test_size=0.2,
        random_state=SEED,
    )

    train_history = (
        train_df.groupby("user_idx")["item_idx"]
        .apply(lambda values: set(int(item) for item in values))
        .to_dict()
    )
    for user_idx in range(data.n_users):
        train_history.setdefault(user_idx, set())

    relevant_holdout = holdout_df[holdout_df["rating"] >= 4]
    ground_truth = (
        relevant_holdout.groupby("user_idx")["item_idx"]
        .apply(lambda values: set(int(item) for item in values))
        .to_dict()
    )
    for user_idx in range(data.n_users):
        ground_truth.setdefault(user_idx, set())

    data.user_history = train_history
    eval_users = [user_idx for user_idx in range(data.n_users) if ground_truth[user_idx]]

    limit = os.environ.get("EVAL_USER_LIMIT")
    if limit:
        eval_users = eval_users[: max(0, int(limit))]

    return ground_truth, eval_users


def initialize_shared_models(data):
    """Initialize the MSRS stakeholder models once."""
    print("\n=== Initializing shared stakeholder models ===")
    movie_genres = {
        item_idx: data.items_df.loc[data.idx2item[item_idx], "genres"].split("|")
        for item_idx in range(data.n_items)
    }

    dopm_model = DOPMRecommender()
    dopm_model.fit(data.user_history, movie_genres)

    linker = DatasetLinker()
    serendipity_model = SerendipityModel(data=data, linker=linker)

    fairness_model = FairnessObjective(data)
    fairness_model.fit()

    return dopm_model, serendipity_model, fairness_model


def ensure_recommendation_length(rec_list, fallback_pool, n=N_MAX_REC):
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


def run_msrs_pipeline(data, R_hat, eval_users, dopm_model, serendipity_model, fairness_model):
    """Run the existing PSNR-filtered MSRS NSGA-II pipeline for eval users."""
    print("\n=== [MSRS] Running PSNR-filtered NSGA-II optimization per user ===")
    msrs_recs = {}

    for pos, user_idx in enumerate(eval_users, start=1):
        print(f"  MSRS user {user_idx} ({pos}/{len(eval_users)})", end="\r")
        pure_signal = get_pure_signal(R_hat[user_idx], data.user_history.get(user_idx, set()))
        candidate_pool = pure_signal["item_indices"][:N_CANDIDATE_POOL]
        candidate_scores = np.asarray(pure_signal["scores"][:N_CANDIDATE_POOL], dtype=np.float64)

        if len(candidate_pool) == 0:
            msrs_recs[int(user_idx)] = []
            continue

        rec_size = min(N_MAX_REC, len(candidate_pool))
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
            variations=MSRS_VARIATIONS,
            pool_size=min(MSRS_ROULETTE_POOL_SIZE, len(candidate_pool)),
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

        np.random.seed(SEED + int(user_idx))
        best_list, _ = run_nsga2_optimization(
            user_idx=int(user_idx),
            candidate_pool=candidate_pool,
            ncf_scores=R_hat[user_idx],
            dopm_recommender=dopm_model,
            serendipity_model=serendipity_model,
            fairness_model=fairness_model,
            user_history=list(data.user_history.get(user_idx, set())),
            initial_population_variants=initial_variants,
            N=rec_size,
            pop_size=MSRS_POP_SIZE,
            n_generations=MSRS_GENERATIONS,
        )
        msrs_recs[int(user_idx)] = ensure_recommendation_length(
            best_list,
            candidate_pool,
            N_MAX_REC,
        )

    print(f"\n[MSRS] Complete for {len(eval_users)} users.")
    return msrs_recs





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


def evaluate_model(recommendations, model_name, data, R_hat, ground_truth, eval_users, dopm_model, objectives, fairness_model):
    print(f"\n=== Evaluating {model_name} ===")
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


def save_all_tables(results, tables_dir, k_values):
    rows = []
    for model in ["FAS-MOEA", "MSRS"]:
        for k in k_values:
            row = {"Model": model, "K": k}
            row.update(results[model][k])
            rows.append(row)
    df_all = pd.DataFrame(rows)

    df_all[df_all["K"] == 10].to_csv(
        os.path.join(tables_dir, "table_summary_all_metrics.csv"),
        index=False,
    )

    acc_cols = ["Model", "K", "Precision@K", "Recall@K", "NDCG@K", "MAP@K", "F1@K"]
    df_all[acc_cols].to_csv(os.path.join(tables_dir, "table_accuracy_ranking.csv"), index=False)

    ba_cols = ["Model", "K", "ILD@K", "Novelty@K", "Serendipity@K"]
    df_all[ba_cols].to_csv(os.path.join(tables_dir, "table_consumer_serendipity.csv"), index=False)

    prov_cols = ["Model", "K", "DOPM@K", "LongTailCoverage@K", "GenreCoverage@K", "CatalogCoverage@K"]
    df_all[prov_cols].to_csv(os.path.join(tables_dir, "table_provider_dopm.csv"), index=False)

    fair_cols = ["Model", "K", "FAS_Fairness@K", "GenreEntropy@K", "ProducerFairness@K"]
    df_all[fair_cols].to_csv(os.path.join(tables_dir, "table_system_fairness.csv"), index=False)

    return df_all


def plot_metric_topn(df_all, metric_name, ylabel, title, filename, figs_dir):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"FAS-MOEA": "#2196F3", "MSRS": "#FF9800"}
    styles = {"FAS-MOEA": "-", "MSRS": "--"}
    markers = {"FAS-MOEA": "o", "MSRS": "s"}
    for model in ["FAS-MOEA", "MSRS"]:
        sub = df_all[df_all["Model"] == model].sort_values("K")
        ax.plot(
            sub["K"],
            sub[metric_name],
            color=colors[model],
            linestyle=styles[model],
            marker=markers[model],
            linewidth=2,
            markersize=6,
            label=model,
        )
    ax.set_xlabel("Top-N Recommendations", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, filename), dpi=300, bbox_inches="tight")
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
    colors = {"FAS-MOEA": "#2196F3", "MSRS": "#FF9800"}

    for model in ["FAS-MOEA", "MSRS"]:
        values = [normalized[metric].get(model, 0.0) for metric in metrics]
        values += values[:1]
        ax.plot(angles, values, color=colors[model], linewidth=2.5, label=model)
        ax.fill(angles, values, color=colors[model], alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("K=10 Normalized Metric Radar", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "fig_radar_summary.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_pareto_3d(fasmoea_pareto, figs_dir):
    rows = []
    for user_idx, metrics_list in fasmoea_pareto.items():
        for metrics in metrics_list or []:
            rows.append(
                [
                    float(metrics.get("accuracy", 0.0)),
                    float(metrics.get("fairness", 0.0)),
                    float(metrics.get("serendipity", 0.0)),
                ]
            )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    if rows:
        values = np.asarray(rows, dtype=float)
        scatter = ax.scatter(
            values[:, 0],
            values[:, 1],
            values[:, 2],
            c=values[:, 2],
            cmap="viridis",
            s=24,
            alpha=0.75,
        )
        fig.colorbar(scatter, ax=ax, shrink=0.65, label="Serendipity")
    else:
        ax.text(0.5, 0.5, 0.5, "No Pareto points available", ha="center", va="center")

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Fairness")
    ax.set_zlabel("Serendipity")
    ax.set_title("FAS-MOEA 3D Pareto Front", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "fig_pareto_fasmoea_3d.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_grouped_bar_k10(df_all, figs_dir):
    groups = [
        ("Consumer Accuracy", ["Precision@K", "Recall@K", "NDCG@K", "F1@K"]),
        ("Consumer Beyond-Acc", ["ILD@K", "Novelty@K", "Serendipity@K"]),
        ("Provider", ["DOPM@K", "LongTailCoverage@K"]),
        ("System Fairness", ["FAS_Fairness@K", "GenreEntropy@K", "ProducerFairness@K"]),
    ]
    df_k10 = df_all[df_all["K"] == 10].set_index("Model")
    colors = {"FAS-MOEA": "#2196F3", "MSRS": "#FF9800"}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

    for ax, (title, metrics) in zip(axes, groups):
        x = np.arange(len(metrics))
        width = 0.35
        fas_vals = [float(df_k10.loc["FAS-MOEA", metric]) for metric in metrics]
        msrs_vals = [float(df_k10.loc["MSRS", metric]) for metric in metrics]

        ax.bar(x - 0.5 * width, fas_vals, width, label="FAS-MOEA", color=colors["FAS-MOEA"])
        ax.bar(x + 0.5 * width, msrs_vals, width, label="MSRS", color=colors["MSRS"])
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([metric.replace("@K", "") for metric in metrics], rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.35)

    axes[0].legend(fontsize=10)
    fig.suptitle("K=10 Stakeholder Metric Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "fig_grouped_bar_k10.png"), dpi=300, bbox_inches="tight")
    plt.close()


def generate_all_figures(df_all, fasmoea_pareto, figs_dir):
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
        plot_metric_topn(df_all, metric, ylabel, title, filename, figs_dir)

    plot_radar_summary(df_all, figs_dir)
    plot_pareto_3d(fasmoea_pareto, figs_dir)
    plot_grouped_bar_k10(df_all, figs_dir)


def print_k10_summary(results):
    key_metrics = [
        "Precision@K",
        "Recall@K",
        "NDCG@K",
        "F1@K",
        "ILD@K",
        "Serendipity@K",
        "Novelty@K",
        "DOPM@K",
        "LongTailCoverage@K",
        "FAS_Fairness@K",
        "GenreEntropy@K",
        "ProducerFairness@K",
    ]

    print(f"\n{'Metric':<25} {'FAS-MOEA':>10} {'MSRS':>10}")
    print("-" * 50)
    for metric in key_metrics:
        v_fas = results["FAS-MOEA"][10].get(metric, 0.0)
        v_msrs = results["MSRS"][10].get(metric, 0.0)
        print(f"{metric:<25} {v_fas:>10.4f} {v_msrs:>10.4f}")


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    start_all = time.perf_counter()
    prepare_results_dirs()

    print("=== Loading MovieLens data ===")
    data = MovieLensData()
    ground_truth, eval_users = build_ground_truth_and_train_history(data)
    if not eval_users:
        raise RuntimeError("No evaluation users with relevant held-out items were found.")
    print(f"Evaluation users: {len(eval_users)}")

    print("\n=== Loading saved NCF model and generating shared R_hat ===")
    model = load_trained_model(data)
    start = time.perf_counter()
    R_hat = generate_full_r_hat(model, data.n_users, data.n_items)
    print(f"Generated R_hat with shape {R_hat.shape} in {time.perf_counter() - start:.2f}s")

    dopm_model, serendipity_model, fairness_model = initialize_shared_models(data)

    # 1. Load or run MSRS
    msrs_pred_path = os.path.join(PREDICTIONS_DIR, "msrs_recommendations.json")
    if os.path.exists(msrs_pred_path):
        print(f"\n[MSRS] Loading existing predictions from {msrs_pred_path}")
        with open(msrs_pred_path, "r", encoding="utf-8") as f:
            msrs_recs = json.load(f)
            msrs_recs = {int(k): v for k, v in msrs_recs.items()}
    else:
        msrs_recs = run_msrs_pipeline(
            data=data,
            R_hat=R_hat,
            eval_users=eval_users,
            dopm_model=dopm_model,
            serendipity_model=serendipity_model,
            fairness_model=fairness_model,
        )
        save_json(msrs_pred_path, msrs_recs)

    # 4. Load or run FAS-MOEA
    fas_pred_path = os.path.join(PREDICTIONS_DIR, "fasmoea_recommendations.json")
    fas_pareto_path = os.path.join(PREDICTIONS_DIR, "fasmoea_pareto.json")
    if os.path.exists(fas_pred_path) and os.path.exists(fas_pareto_path):
        print(f"\n[FAS-MOEA] Loading existing predictions from {fas_pred_path}")
        with open(fas_pred_path, "r", encoding="utf-8") as f:
            fasmoea_recs = json.load(f)
            fasmoea_recs = {int(k): v for k, v in fasmoea_recs.items()}
        with open(fas_pareto_path, "r", encoding="utf-8") as f:
            fasmoea_pareto = json.load(f)
            fasmoea_pareto = {int(k): v for k, v in fasmoea_pareto.items()}
        from fasmoea.fasmoea_model import FASMOEAObjectives
        fasmoea_objectives = FASMOEAObjectives(data, R_hat).fit()
    else:
        fasmoea_recs, fasmoea_pareto, fasmoea_objectives = run_fasmoea_pipeline(data, R_hat, eval_users)
        save_json(fas_pred_path, fasmoea_recs)
        save_json(fas_pareto_path, fasmoea_pareto)

    results = {
        "FAS-MOEA": evaluate_model(
            fasmoea_recs,
            "FAS-MOEA",
            data,
            R_hat,
            ground_truth,
            eval_users,
            dopm_model,
            fasmoea_objectives,
            fairness_model,
        ),
        "MSRS": evaluate_model(
            msrs_recs,
            "MSRS",
            data,
            R_hat,
            ground_truth,
            eval_users,
            dopm_model,
            fasmoea_objectives,
            fairness_model,
        ),
    }

    df_all = save_all_tables(results, TABLES_DIR, K_VALUES)
    generate_all_figures(df_all, fasmoea_pareto, FIGURES_DIR)
    print_k10_summary(results)

    print(f"\nSaved predictions to {PREDICTIONS_DIR}")
    print(f"Saved tables to {TABLES_DIR}")
    print(f"Saved figures to {FIGURES_DIR}")
    print(f"Total runtime: {time.perf_counter() - start_all:.2f}s")


if __name__ == "__main__":
    main()
