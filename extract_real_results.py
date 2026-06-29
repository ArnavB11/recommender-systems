import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV

from data_loader import MovieLensData
from ncf_inference import get_pure_signal, load_trained_model
from serendipity import SerendipityModel
from dataset_linker import DatasetLinker
from dopm import DOPMRecommender
from fairness import FairnessObjective
from genetic_algorithm import RecommendationListProblem, OrderCrossover, SwapMutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.population import Population
from weighted_roulette import generate_weighted_roulette_lists
from psnr_filter import filter_by_psnr_sweetspot

# Set up seed for reproducibility
np.random.seed(42)

# ==========================================
# Constants matching main.py & genetic_algorithm.py
# ==========================================
N_LIST_SIZE = 10
POP_SIZE = 50
N_GENERATIONS = 500

# ==========================================
# 1. DEFINE LOGGING CALLBACK FOR PYMOO
# ==========================================

class NSGALogger(Callback):
    """Logs the Pareto front at every generation for HV and IGD computation."""
    def __init__(self, ref_point):
        super().__init__()
        self.ref_point = ref_point
        self.hv_history = []
        self.fronts = []
        
    def notify(self, algorithm):
        opt_F = algorithm.opt.get("F")
        self.fronts.append(opt_F.copy())
        
        hv_indicator = HV(ref_point=self.ref_point)
        hv_val = hv_indicator.do(opt_F)
        self.hv_history.append(float(hv_val))

# ==========================================
# 2. RUN REAL PIPELINE (MATCHING main.py)
# ==========================================

def run_real_pipeline():
    print("=" * 75)
    print("REAL DATA EXTRACTION & ANALYSIS PIPELINE")
    print(f"Using N_LIST_SIZE={N_LIST_SIZE}, POP_SIZE={POP_SIZE}, N_GENERATIONS={N_GENERATIONS}")
    print("=" * 75)
    
    # --- Phase 1: Load Data & NCF Model (identical to main.py) ---
    print("\n--- Phase 1: Loading Real Dataset and NCF Model ---")
    data = MovieLensData()
    model = load_trained_model(data)
    
    # Generate full R_hat for all users
    all_users = torch.arange(data.n_users).repeat_interleave(data.n_items)
    all_items = torch.arange(data.n_items).repeat(data.n_users)
    with torch.no_grad():
        all_scores = model(all_users, all_items)
    R_hat = all_scores.detach().cpu().numpy().reshape(data.n_users, data.n_items)
    print(f"Generated R_hat with shape {R_hat.shape}")
    
    # --- Phase 2: Initialize Models (identical to main.py) ---
    print("\n--- Phase 2: Initializing DOPM, BERT Serendipity & Fairness Models ---")
    dopm_system = DOPMRecommender()
    movie_genres = {
        item_idx: data.items_df.loc[data.idx2item[item_idx], "genres"].split("|")
        for item_idx in range(data.n_items)
    }
    dopm_system.fit(data.user_history, movie_genres)
    
    linker = DatasetLinker()
    serendipity_system = SerendipityModel(data=data, linker=linker)
    
    fairness_model = FairnessObjective(data)
    fairness_model.fit()
    
    # --- Phase 3: Build P0 for a sample user (identical to main.py Phase 3) ---
    user_idx = 0
    print(f"\n--- Phase 3: Building P0 for User {user_idx} (identical to main.py) ---")
    
    pure_signal = get_pure_signal(R_hat[user_idx], data.user_history.get(user_idx, set()))
    candidate_items = pure_signal["item_indices"]
    candidate_scores = np.asarray(pure_signal["scores"], dtype=np.float64)
    
    movie_scores_dict = {
        item_idx: float(score)
        for item_idx, score in zip(candidate_items, candidate_scores)
    }
    
    roulette_lists = generate_weighted_roulette_lists(
        pure_signal_scores=candidate_scores.reshape(1, -1),
        user_watch_matrix=np.zeros((1, len(candidate_items)), dtype=np.int8),
        users=[user_idx],
        movies=candidate_items,
        N=N_LIST_SIZE,
        variations=1000,
        pool_size=50,
    )
    
    original_ncf_list = candidate_items[:N_LIST_SIZE]
    accepted_variants, low, high = filter_by_psnr_sweetspot(
        original_list=original_ncf_list,
        variant_lists=roulette_lists[user_idx],
        movie_scores_dict=movie_scores_dict,
        lower_percentile=25,
        upper_percentile=75,
        max_val=1.0,
    )
    
    initial_population_variants = [variant["list"] for variant in accepted_variants]
    user_history = list(data.user_history.get(user_idx, set()))
    
    print(f"Built P0 with {len(initial_population_variants)} accepted variants")
    print(f"Candidate pool size: {len(candidate_items)}")
    
    # --- Phase 4: Run NSGA-II with per-generation logging ---
    # We run NSGA-II manually (not via run_nsga2_optimization) so we can
    # attach our NSGALogger callback to capture per-generation metrics.
    
    def run_nsga2_with_logging(label, candidate_pool, initial_variants):
        """Run NSGA-II with generation-level HV/IGD logging."""
        pool_size = len(candidate_pool)
        
        problem = RecommendationListProblem(
            user_idx=user_idx,
            candidate_pool=candidate_pool,
            ncf_scores=R_hat[user_idx],
            dopm_recommender=dopm_system,
            serendipity_model=serendipity_system,
            fairness_model=fairness_model,
            user_history=user_history,
            N=N_LIST_SIZE
        )
        
        # Build initial population from P0 (identical to genetic_algorithm.py)
        movie_to_pool_idx = {movie: idx for idx, movie in enumerate(candidate_pool)}
        initial_chromosomes = []
        
        for variant in initial_variants:
            mapped = [movie_to_pool_idx[m] for m in variant if m in movie_to_pool_idx]
            if len(mapped) < N_LIST_SIZE:
                remaining = list(set(range(pool_size)) - set(mapped))
                mapped += list(np.random.choice(remaining, size=N_LIST_SIZE - len(mapped), replace=False))
            else:
                mapped = mapped[:N_LIST_SIZE]
            initial_chromosomes.append(mapped)
        
        while len(initial_chromosomes) < POP_SIZE:
            initial_chromosomes.append(list(np.random.choice(pool_size, size=N_LIST_SIZE, replace=False)))
        
        initial_chromosomes = np.array(initial_chromosomes)
        pop = Population.new("X", initial_chromosomes)
        
        algorithm = NSGA2(
            pop_size=POP_SIZE,
            sampling=pop,
            crossover=OrderCrossover(prob=0.9),
            mutation=SwapMutation(prob=1.0/N_LIST_SIZE),
            eliminate_duplicates=True
        )
        
        # Reference point for HV: since we minimize [-DOPM, -Ser, -Fairness],
        # the ref point should be the worst case (all zeros → negated = 0.0)
        logger = NSGALogger(ref_point=np.array([0.0, 0.0, 0.0]))
        
        print(f"Running NSGA-II for {label} ({N_GENERATIONS} Generations, pop_size={POP_SIZE})...")
        start = time.perf_counter()
        res = minimize(
            problem,
            algorithm,
            termination=('n_gen', N_GENERATIONS),
            callback=logger,
            seed=42,
            verbose=False
        )
        elapsed = time.perf_counter() - start
        print(f"{label} optimization completed in {elapsed:.2f} seconds.")
        
        return res, logger, candidate_pool
    
    # Run the optimization for the Proposed FAS-MOEA
    print("\n--- Phase 4: Optimizing Proposed FAS-MOEA ---")
    res, logger, pool = run_nsga2_with_logging(
        "Proposed FAS-MOEA", candidate_items, initial_population_variants
    )
    
    # ==========================================
    # 3. EXTRACT REAL METRICS & BASELINE COMPARISON
    # ==========================================
    print("\n--- Phase 5: Extracting Real Metrics and Computing NCF Baseline ---")
    
    user_pref_vec = serendipity_system.compute_user_preference_vector(set(user_history))
    
    def evaluate_list(movie_list):
        dopm_scores = [
            dopm_system.calculate_dopm(user_idx, m, R_hat[user_idx][m])
            for m in movie_list
        ]
        ser_scores = [
            serendipity_system.calculate_serendipity(user_history, user_pref_vec, m)
            for m in movie_list
        ]
        dopm_dict = {m: score for m, score in zip(movie_list, dopm_scores)}
        ser_dict = {m: score for m, score in zip(movie_list, ser_scores)}
        fair = fairness_model.compute_fairness(movie_list, dopm_dict, ser_dict)
        return np.mean(dopm_scores), np.mean(ser_scores), fair

    def calculate_mohs(dopm, ser, fair):
        dopm = max(dopm, 1e-6)
        ser = max(ser, 1e-6)
        fair = max(fair, 1e-6)
        return 3.0 / (1.0 / dopm + 1.0 / ser + 1.0 / fair)

    # 3.1 Evaluate Baseline NCF List
    base_dopm, base_ser, base_fair = evaluate_list(original_ncf_list)
    base_mohs = calculate_mohs(base_dopm, base_ser, base_fair)
    
    # 3.2 Extract entire Pareto front points from optimization
    pareto_points = []
    
    for chromosome in res.X.astype(int):
        movie_list = [pool[idx] for idx in chromosome]
        dopm_val, ser_val, fair_val = evaluate_list(movie_list)
        mohs_val = calculate_mohs(dopm_val, ser_val, fair_val)
        pareto_points.append([dopm_val, ser_val, fair_val, mohs_val])
        
    pareto_points = np.array(pareto_points)
    
    # 3.3 Find the best compromise solution from Pareto Front (maximizes MOHS)
    best_idx = np.argmax(pareto_points[:, 3])
    comp_dopm, comp_ser, comp_fair, comp_mohs = pareto_points[best_idx]
    
    print("\n" + "="*75)
    print("ALGORITHM INSIGHTS & RESULTS SUMMARY")
    print("="*75)
    print(f"{'Metric':<25} | {'Baseline NCF':<20} | {'Proposed FAS-MOEA':<20} | {'Net Change':<12}")
    print("-"*75)
    print(f"{'Personalized Genre Novelty':<25} | {base_dopm:<20.4f} | {comp_dopm:<20.4f} | {((comp_dopm - base_dopm)/base_dopm * 100):+7.2f}%")
    print(f"{'BERT Serendipity':<25} | {base_ser:<20.4f} | {comp_ser:<20.4f} | {((comp_ser - base_ser)/base_ser * 100):+7.2f}%")
    print(f"{'Provider Fairness':<25} | {base_fair:<20.4f} | {comp_fair:<20.4f} | {((comp_fair - base_fair)/base_fair * 100):+7.2f}%")
    print("-"*75)
    print(f"{'MOHS Quality Score':<25} | {base_mohs:<20.4f} | {comp_mohs:<20.4f} | {((comp_mohs - base_mohs)/base_mohs * 100):+7.2f}%")
    print("="*75)
    
    # 3.4 Calculate IGD convergence curve
    # The reference Pareto front P* is approximated by the absolute best non-dominated front 
    # found across all generations of optimization
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    all_fronts_combined = np.vstack(logger.fronts)
    nds = NonDominatedSorting()
    front_indices = nds.do(all_fronts_combined, only_non_dominated_front=True)
    P_star = all_fronts_combined[front_indices]
    
    def calculate_igd(front_F, reference_F):
        distances = []
        for v in reference_F:
            d = np.min(np.linalg.norm(front_F - v, axis=1))
            distances.append(d)
        return float(np.mean(distances))
        
    igd_history = [calculate_igd(f, P_star) for f in logger.fronts]
    
    # 3.5 Calculate baseline NCF's static hypervolume
    # NCF baseline represents a single point in the objective space: [-base_dopm, -base_ser, -base_fair]
    base_F = np.array([[-base_dopm, -base_ser, -base_fair]])
    hv_indicator = HV(ref_point=np.array([0.0, 0.0, 0.0]))
    base_hv = hv_indicator.do(base_F)
    
    # ==========================================
    # 4. PLOT EXACTLY 4 HIGHLY RELEVANT INSIGHTFUL GRAPHS
    # ==========================================
    print("\n--- Phase 6: Plotting Publication-Quality Graphs with Real Data ---")
    
    artifacts_dir = r"C:\Users\bhagw\.gemini\antigravity-ide\brain\69f3e31f-50f3-488d-bb73-9ad30a523b08"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
    plt.rcParams['text.usetex'] = False
    plt.rcParams['axes.unicode_minus'] = False
    
    # Academic Color Palette
    CRIMSON = '#DC143C'
    NAVY = '#1A237E'
    GOLD = '#FFD700'
    CHARCOAL = '#333333'
    
    # -------------------------------------------------------------------------
    # GRAPH 1: 3D Pareto Optimal Frontier with NCF Baseline Point
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 8), dpi=300, facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    
    # Clean background panes to white
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    
    grid_style = {'color': (0.85, 0.85, 0.85, 1.0), 'linewidth': 0.5, 'linestyle': ':'}
    ax.xaxis._axinfo["grid"].update(grid_style)
    ax.yaxis._axinfo["grid"].update(grid_style)
    ax.zaxis._axinfo["grid"].update(grid_style)
    
    # Scatter plot of Pareto front solutions
    ax.scatter(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2], 
               color=NAVY, s=40, edgecolors='white', linewidths=0.4, alpha=0.8, 
               label='Proposed FAS-MOEA Pareto Front')
               
    # Plot baseline NCF as a prominent gold star
    ax.scatter([base_dopm], [base_ser], [base_fair], 
               color=CRIMSON, s=250, marker='*', edgecolors='black', linewidths=1.2, zorder=100,
               label='Baseline NCF')
               
    # Highlight the compromise solution chosen by MOHS
    ax.scatter([comp_dopm], [comp_ser], [comp_fair],
               color=GOLD, s=120, marker='o', edgecolors=CHARCOAL, linewidths=1.2, zorder=90,
               label='FAS-MOEA Best Compromise (Max MOHS)')
    
    ax.set_xlabel('Personalized Genre Novelty (DOPM)', fontsize=11, labelpad=12, fontweight='semibold')
    ax.set_ylabel('BERT Semantic Serendipity', fontsize=11, labelpad=12, fontweight='semibold')
    ax.set_zlabel('Provider Exposure Fairness', fontsize=11, labelpad=12, fontweight='semibold')
    
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.view_init(elev=20, azim=-60)
    
    plt.title("Figure 1: 3D Pareto Optimal Frontier vs. NCF Baseline", 
              fontsize=13, pad=15, fontweight='bold', color=CHARCOAL)
    
    legend = ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.85), frameon=True, 
                       facecolor='white', edgecolor='#e0e0e0', fontsize=9.5)
    legend.get_frame().set_linewidth(0.8)
    plt.tight_layout()
    
    plt.savefig('pareto_3d_frontier.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(artifacts_dir, 'pareto_3d_frontier.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved Graph 1: pareto_3d_frontier.png")
    
    # -------------------------------------------------------------------------
    # GRAPH 2: Hypervolume (HV) Convergence Curve
    # -------------------------------------------------------------------------
    generations = np.arange(1, N_GENERATIONS + 1)
    marker_interval = 50  # Marker every 50 generations
    
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300, facecolor='white')
    ax.set_facecolor('white')
    ax.grid(True, which='both', color='#e0e0e0', linestyle='--', linewidth=0.5)
    
    # Plot FAS-MOEA Hypervolume curve
    ax.plot(generations, logger.hv_history, color=NAVY, linestyle='-', linewidth=2.2,
            marker='o', markersize=6, markevery=marker_interval, markerfacecolor='white',
            markeredgewidth=1.5, markeredgecolor=NAVY,
            label='Proposed FAS-MOEA (Evolving Pareto Front)')
            
    # Plot NCF Baseline static Hypervolume as a horizontal dashed line
    ax.axhline(y=base_hv, color=CRIMSON, linestyle='--', linewidth=1.8,
               label=f'Baseline NCF (Static Single List, HV = {base_hv:.4f})')
            
    ax.set_xlabel('Generations', fontsize=12, fontweight='semibold', labelpad=10)
    ax.set_ylabel('Hypervolume (HV) Metric', fontsize=12, fontweight='semibold', labelpad=10)
    
    ax.set_xlim(0, N_GENERATIONS + 5)
    ax.set_xticks(np.arange(0, N_GENERATIONS + 1, 50))
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.title(f"Figure 2: Hypervolume Convergence Curve over {N_GENERATIONS} Generations", 
              fontsize=13, pad=15, fontweight='bold', color=CHARCOAL)
              
    legend = ax.legend(loc='lower right', frameon=True, facecolor='white', 
                       edgecolor='#d0d0d0', fontsize=10.5)
    legend.get_frame().set_linewidth(0.8)
    plt.tight_layout()
    
    plt.savefig('visualize_convergence.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(artifacts_dir, 'visualize_convergence.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved Graph 2: visualize_convergence.png")
    
    # -------------------------------------------------------------------------
    # GRAPH 3: IGD Convergence Error Curve
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300, facecolor='white')
    ax.set_facecolor('white')
    ax.grid(True, which='both', color='#e0e0e0', linestyle='--', linewidth=0.5)
    
    # Plot IGD Error Curve
    ax.plot(generations, igd_history, color=NAVY, linestyle='-', linewidth=2.2,
            marker='d', markersize=6, markevery=marker_interval, markerfacecolor='white',
            markeredgewidth=1.5, markeredgecolor=NAVY,
            label='Inverted Generational Distance (IGD)')
            
    ax.set_xlabel('Generations', fontsize=12, fontweight='semibold', labelpad=10)
    ax.set_ylabel('IGD Error (Convergence Indicator)', fontsize=12, fontweight='semibold', labelpad=10)
    
    ax.set_xlim(0, N_GENERATIONS + 5)
    ax.set_xticks(np.arange(0, N_GENERATIONS + 1, 50))
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.title(f"Figure 3: IGD Convergence Error Curve over {N_GENERATIONS} Generations", 
              fontsize=13, pad=15, fontweight='bold', color=CHARCOAL)
              
    legend = ax.legend(loc='upper right', frameon=True, facecolor='white', 
                       edgecolor='#d0d0d0', fontsize=11)
    legend.get_frame().set_linewidth(0.8)
    plt.tight_layout()
    
    plt.savefig('visualize_igd.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(artifacts_dir, 'visualize_igd.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved Graph 3: visualize_igd.png")
    
    # -------------------------------------------------------------------------
    # GRAPH 4: Grouped Bar Chart Comparison of NCF Baseline vs Compromise Solution
    # -------------------------------------------------------------------------
    metrics = ['Personalized Genre\nNovelty (DOPM)', 'BERT\nSerendipity', 'Provider\nFairness', 'MOHS\n(Unified Score)']
    baseline_vals = [base_dopm, base_ser, base_fair, base_mohs]
    compromise_vals = [comp_dopm, comp_ser, comp_fair, comp_mohs]
    
    x = np.arange(len(metrics))
    width = 0.35  # width of the bars
    
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300, facecolor='white')
    ax.set_facecolor('white')
    ax.grid(True, axis='y', which='major', color='#e0e0e0', linestyle='--', linewidth=0.5, zorder=0)
    
    rects1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline NCF', color=CRIMSON, edgecolor='black', linewidth=0.7, zorder=3)
    rects2 = ax.bar(x + width/2, compromise_vals, width, label='Proposed FAS-MOEA (Compromise)', color=NAVY, edgecolor='black', linewidth=0.7, zorder=3)
    
    # Label styling
    ax.set_ylabel('Objective Metric Value', fontsize=12, fontweight='semibold', labelpad=10)
    ax.set_title('Figure 4: Direct Metric Comparison & MOHS Quality Net Gain', fontsize=13, pad=20, fontweight='bold', color=CHARCOAL)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight='semibold')
    ax.tick_params(axis='both', which='major', labelsize=10.5)
    
    # Legend
    legend = ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#d0d0d0', fontsize=11)
    legend.get_frame().set_linewidth(0.8)
    
    # Add value labels on top of the bars with percentage improvements
    def autolabel(rects, base_rects=None):
        for idx, rect in enumerate(rects):
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9.5, fontweight='semibold')
                        
            # If it's the compromise bars, display the net improvement on top
            if base_rects is not None:
                base_height = base_rects[idx].get_height()
                improvement = ((height - base_height) / base_height) * 100
                color = 'green' if improvement >= 0 else 'red'
                ax.annotate(f'{improvement:+.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 15),  # 15 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)

    autolabel(rects1)
    autolabel(rects2, rects1)
    
    # Add buffer on top for text labels
    ax.set_ylim(0, max(max(baseline_vals), max(compromise_vals)) * 1.25)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(artifacts_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved Graph 4: performance_comparison.png")
    
    print("\n" + "=" * 75)
    print("ALL 4 CRITICAL INSIGHT GRAPHS GENERATED SUCCESSFULLY WITH REAL DATA!")
    print("=" * 75)

if __name__ == "__main__":
    run_real_pipeline()
