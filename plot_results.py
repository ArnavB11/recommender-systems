import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_pareto(data):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    all_acc, all_fair, all_ser = [], [], []
    for user_front in data["pareto_fronts"]:
        for sol in user_front:
            all_acc.append(sol["accuracy"])
            all_fair.append(sol["fairness"])
            all_ser.append(sol["serendipity"])
            
    # Scatter plot matching the research paper style
    ax.scatter(all_acc, all_fair, all_ser, c='tab:blue', s=40, alpha=0.8, edgecolors='w', linewidth=0.5)
    
    ax.set_xlabel('Accuracy (NCF Score Proxy)')
    ax.set_ylabel('Fairness')
    ax.set_zlabel('Serendipity')
    ax.set_title('NSGA-II Pareto Front (MovieLens)')
    
    # View angle tuning
    ax.view_init(elev=20, azim=45)
    
    plt.savefig('results/pareto_front.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_hypervolume(data):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Pad shorter HV histories with their last value to compute the mean
    hv_data = data.get("hypervolume", [])
    if not hv_data:
        return
        
    max_len = max(len(h) for h in hv_data if len(h) > 0)
    
    padded_hv = []
    for h in hv_data:
        if len(h) == 0:
            continue
        padded = list(h) + [h[-1]] * (max_len - len(h))
        padded_hv.append(padded)
        
    padded_igd = []
    for ig in data.get("igd", []):
        if len(ig) == 0:
            continue
        padded = list(ig) + [ig[-1]] * (max_len - len(ig))
        padded_igd.append(padded)
        
    if not padded_hv or not padded_igd:
        return
        
    mean_hv = np.mean(padded_hv, axis=0)
    mean_igd = np.mean(padded_igd, axis=0)
    generations = np.arange(1, max_len + 1)
    
    # Left Y-axis (Hypervolume)
    color1 = 'tab:blue'
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Metric Value (Hypervolume)', color=color1)
    line1 = ax1.plot(generations, mean_hv, linestyle='-', color=color1, linewidth=2.5, label='Hypervolume')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Right Y-axis (IGD)
    ax2 = ax1.twinx()  
    color2 = 'tab:orange'
    ax2.set_ylabel('Metric Value (IGD)', color=color2)
    line2 = ax2.plot(generations, mean_igd, linestyle='-', color=color2, linewidth=2.5, label='IGD')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Legend
    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center left')

    plt.title('Convergence of FAS-MOEA on MovieLens')
    
    fig.tight_layout()
    plt.savefig('results/hypervolume.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_top_n_metrics(data):
    metrics_group_1 = ["precision", "recall", "ndcg", "map", "f1"]
    metrics_group_2 = ["diversity", "explainability", "novelty", "fairness", "serendipity"]
    
    n_values = sorted([int(k) for k in data["top_n_metrics"]["ncf"].keys()])
    
    def plot_group(metrics, filename):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            ncf_vals = [data["top_n_metrics"]["ncf"][str(k)][metric] for k in n_values]
            nsga2_vals = [data["top_n_metrics"]["nsga2"][str(k)][metric] for k in n_values]
            
            ax.plot(n_values, ncf_vals, marker='o', label='NCF (Baseline)', color='tab:gray')
            ax.plot(n_values, nsga2_vals, marker='^', label='NSGA-II (Proposed)', color='tab:red')
            
            ax.set_title(f'{metric.capitalize()}@t vs Top-n')
            ax.set_xlabel('Top-n')
            ax.set_ylabel(metric.capitalize())
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            
        # Hide the last empty subplot if odd number
        if len(metrics) < len(axes):
            for j in range(len(metrics), len(axes)):
                axes[j].set_visible(False)
                
        plt.tight_layout()
        plt.savefig(f'results/{filename}.png', dpi=300)
        plt.close()
        
    plot_group(metrics_group_1, 'top_n_accuracy')
    plot_group(metrics_group_2, 'top_n_diversity')

def plot_percentage_improvement(data):
    n_values = sorted([int(k) for k in data["top_n_metrics"]["ncf"].keys()])
    
    # Strictly the three objective functions (using NDCG to represent DOPM/Accuracy)
    metrics = ["ndcg", "serendipity", "fairness"]
    improvements = {}
    
    for m in metrics:
        ncf_vals = [data["top_n_metrics"]["ncf"][str(k)][m] for k in n_values]
        nsga2_vals = [data["top_n_metrics"]["nsga2"][str(k)][m] for k in n_values]
        
        ncf_mean = np.mean(ncf_vals)
        nsga2_mean = np.mean(nsga2_vals)
        
        # Calculate percentage improvement
        if ncf_mean > 0:
            imp = ((nsga2_mean - ncf_mean) / ncf_mean) * 100
        elif ncf_mean == 0 and nsga2_mean > 0:
            imp = 100.0 
        else:
            imp = 0.0
            
        # Map names exactly to the 3 objective functions
        if m == 'ndcg':
            label = 'DOPM'
        else:
            label = m.capitalize()
            
        improvements[label] = imp

    # Overall average improvement score across the 3 objectives
    overall = np.mean(list(improvements.values()))
    improvements["OVERALL SCORE"] = overall
    
    # Generate Bar Chart
    fig, ax = plt.subplots(figsize=(9, 6))
    names = list(improvements.keys())
    values = list(improvements.values())
    
    colors = ['tab:green' if v >= 0 else 'tab:red' for v in values]
    bars = ax.bar(names, values, color=colors, edgecolor='black', alpha=0.8)
    
    ax.axhline(0, color='black', linewidth=1.2)
    ax.set_ylabel('Percentage Improvement (%)', fontsize=12)
    ax.set_title('NSGA-II vs NCF Baseline: % Improvement of Objective Functions', fontsize=14)
    
    # Add value labels above/below bars
    for bar in bars:
        yval = bar.get_height()
        offset = max(2.0, abs(yval) * 0.05)
        ax.text(bar.get_x() + bar.get_width()/2, 
                yval + offset if yval >= 0 else yval - offset - 2, 
                f"{yval:.1f}%", ha='center', va='bottom' if yval >= 0 else 'top', 
                fontweight='bold', color='black', fontsize=11)
                
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/improvement_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_tables(data):
    print("\n" + "="*50)
    print("Table 5: Runtime and Memory (MovieLens sample)")
    print("="*50)
    mean_runtime = np.mean(data["runtime"])
    mean_mem = np.mean(data["memory"])
    print(f"Proposed Algorithm Runtime (avg per user): {mean_runtime:.2f} s")
    print(f"Proposed Algorithm Peak Memory: {mean_mem:.2f} MB")
    print("="*50 + "\n")

if __name__ == "__main__":
    with open("results/experiment_data.json", "r") as f:
        data = json.load(f)
        
    plot_pareto(data)
    plot_hypervolume(data)
    plot_top_n_metrics(data)
    plot_percentage_improvement(data)
    print_tables(data)
    print("Plots generated successfully in results/ directory.")
