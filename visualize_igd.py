import os
import numpy as np
import matplotlib.pyplot as plt

# Set up seed for reproducibility
np.random.seed(42)

# ==========================================
# 1. GENERATE SYNTHETIC IGD CONVERGENCE DATA
# ==========================================

# X-axis: Generations from 0 to 500 in steps of 10
generations = np.arange(0, 501, 10)
n_points = len(generations)

def generate_igd_curve(L_target, tau, initial_val=0.28, noise_scale=0.002):
    """
    Generates a realistic evolutionary IGD (Inverted Generational Distance) curve.
    Uses an exponential decay model: IGD(g) = L + (initial_val - L) * e^(-g/tau) + noise
    Ensures the curve is mathematically non-increasing (representing archive IGD).
    """
    # Base exponential decay
    igd = L_target + (initial_val - L_target) * np.exp(-generations / tau)
    
    # Add tiny random walks (fluctuations)
    noise = np.random.normal(0, noise_scale, n_points)
    # Cumulative noise subtraction ensures steady convergence downward
    noise_cumulative = np.cumsum(np.abs(noise) * 0.15)
    igd_noisy = igd - noise_cumulative
    
    # Smooth the curve and enforce non-increasing property of archive IGD
    igd_mono = np.minimum.accumulate(igd_noisy)
    
    # Scale to perfectly fit target bounds [initial_val down to L_target]
    igd_final = L_target + (initial_val - L_target) * (igd_mono - igd_mono[-1]) / (igd_mono[0] - igd_mono[-1])
    
    return igd_final

# Generate curves for both algorithms
# User-CF plateaus around 0.025 (lower error), Item-CF plateaus around 0.045
igd_user = generate_igd_curve(L_target=0.025, tau=38.0, initial_val=0.28)
igd_item = generate_igd_curve(L_target=0.045, tau=34.0, initial_val=0.28)

# ==========================================
# 2. PLOTTING SETUP & CUSTOM ACADEMIC STYLE
# ==========================================

# Configure professional font size and family for top-tier journals (ACM/IEEE)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
plt.rcParams['text.usetex'] = False
plt.rcParams['axes.unicode_minus'] = False

# Create figure
fig, ax = plt.subplots(figsize=(9, 6), dpi=300, facecolor='white')
ax.set_facecolor('white')

# Enable crisp, light-grey dashed gridlines
ax.grid(True, which='both', color='#e0e0e0', linestyle='--', linewidth=0.5)

# Colors: High contrast Dark Blue and Dark Orange
DARK_BLUE = '#0D47A1'
DARK_ORANGE = '#E65100'

# Plot lines with markers at 50 generation intervals (every 5th point)
marker_interval = 5

ax.plot(generations, igd_user, color=DARK_BLUE, linestyle='-', linewidth=2.0,
        marker='o', markersize=6, markevery=marker_interval, markerfacecolor='white',
        markeredgewidth=1.5, markeredgecolor=DARK_BLUE,
        label='Proposed FAS-MOEA (User-CF)')

ax.plot(generations, igd_item, color=DARK_ORANGE, linestyle='--', linewidth=2.0,
        marker='d', markersize=6, markevery=marker_interval, markerfacecolor='white',
        markeredgewidth=1.5, markeredgecolor=DARK_ORANGE,
        label='Proposed FAS-MOEA (Item-CF)')

# ==========================================
# 3. LABELS, LEGEND, TITLE & AXIS STYLING
# ==========================================

# Label configurations
ax.set_xlabel('Generations', fontsize=12, fontweight='semibold', labelpad=10)
ax.set_ylabel('Inverted Generational Distance (IGD)', fontsize=12, fontweight='semibold', labelpad=10)

# Configure ticks and limits
ax.set_xlim(-10, 510)
ax.set_ylim(0.00, 0.32)
ax.set_xticks(np.arange(0, 501, 50))
ax.set_yticks(np.arange(0.00, 0.33, 0.05))
ax.tick_params(axis='both', which='major', labelsize=10.5)

# Title (academic figure caption style)
plt.title("Figure 5: IGD Convergence Error Curve across 500 Generations", 
          fontsize=13, pad=15, fontweight='bold')

# Elegant legend with white background and subtle grey border in the top-right
legend = ax.legend(loc='upper right', frameon=True, facecolor='white', 
                   edgecolor='#d0d0d0', fontsize=11, shadow=False)
legend.get_frame().set_linewidth(0.8)

plt.tight_layout()

# Save paths
workspace_img_path = 'visualize_igd.png'
plt.savefig(workspace_img_path, dpi=300, bbox_inches='tight', facecolor='white')

# Save to the artifacts directory
artifacts_dir = r"C:\Users\bhagw\.gemini\antigravity-ide\brain\69f3e31f-50f3-488d-bb73-9ad30a523b08"
os.makedirs(artifacts_dir, exist_ok=True)
artifacts_img_path = os.path.join(artifacts_dir, 'visualize_igd.png')
plt.savefig(artifacts_img_path, dpi=300, bbox_inches='tight', facecolor='white')

print(f"Plot successfully saved to artifacts directory: {artifacts_img_path}")
print(f"Plot successfully saved locally: {workspace_img_path}")
plt.close()
