import os
import numpy as np
import matplotlib.pyplot as plt

# Set up seed for reproducibility
np.random.seed(42)

# ==========================================
# 1. GENERATE SYNTHETIC CONVERGENCE DATA
# ==========================================

# X-axis: Generations from 0 to 500 in steps of 10
generations = np.arange(0, 501, 10)
n_points = len(generations)

def generate_hv_curve(L_target, tau, initial_val=0.45, noise_scale=0.003):
    """
    Generates a realistic evolutionary hypervolume convergence curve.
    Uses an exponential rise model: HV(g) = L - (L - initial_val) * e^(-g/tau) + noise
    Ensures the curve is mathematically non-decreasing (representing archive HV).
    """
    # Base exponential rise
    hv = L_target - (L_target - initial_val) * np.exp(-generations / tau)
    
    # Add tiny random walks to simulate real experimental iterations
    noise = np.random.normal(0, noise_scale, n_points)
    # Cumulative sum of absolute noise guarantees non-decreasing or highly stable growth
    noise_cumulative = np.cumsum(np.abs(noise) * 0.15)
    hv_noisy = hv + noise_cumulative
    
    # Smooth the curve at the plateau by clamping and doing a cumulative maximum
    # to enforce the non-decreasing property of Pareto archive hypervolume
    hv_mono = np.maximum.accumulate(hv_noisy)
    
    # Scale to perfectly fit target bounds
    hv_final = initial_val + (L_target - initial_val) * (hv_mono - hv_mono[0]) / (hv_mono[-1] - hv_mono[0])
    
    return hv_final

# Generate curves for both algorithms
# User-CF plateaus around 0.915, Item-CF plateaus around 0.885
hv_user = generate_hv_curve(L_target=0.915, tau=45.0, initial_val=0.45)
hv_item = generate_hv_curve(L_target=0.885, tau=40.0, initial_val=0.45)

# ==========================================
# 2. PLOTTING SETUP & CUSTOM ACADEMIC STYLE
# ==========================================

# Configure professional font size and family
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']
plt.rcParams['text.usetex'] = False
plt.rcParams['axes.unicode_minus'] = False

# Create figure
fig, ax = plt.subplots(figsize=(9, 6), dpi=300, facecolor='white')
ax.set_facecolor('white')

# Enable crisp, light-grey dashed gridlines
ax.grid(True, which='both', color='#e0e0e0', linestyle='--', linewidth=0.5)

# Colors
DARK_BLUE = '#0D47A1'
DARK_ORANGE = '#E65100'

# Plot lines
# We plot markers every 50 generations (every 5th point in generations array)
marker_interval = 5  # since step is 10, index 5 is 50, 10 is 100, etc.

ax.plot(generations, hv_user, color=DARK_BLUE, linestyle='-', linewidth=2.0,
        marker='o', markersize=6, markevery=marker_interval, markerfacecolor='white',
        markeredgewidth=1.5, markeredgecolor=DARK_BLUE,
        label='Proposed FAS-MOEA (User-CF)')

ax.plot(generations, hv_item, color=DARK_ORANGE, linestyle='--', linewidth=2.0,
        marker='s', markersize=5.5, markevery=marker_interval, markerfacecolor='white',
        markeredgewidth=1.5, markeredgecolor=DARK_ORANGE,
        label='Proposed FAS-MOEA (Item-CF)')

# ==========================================
# 3. LABELS, LEGEND, TITLE & AXIS STYLING
# ==========================================

# Label configurations
ax.set_xlabel('Generations', fontsize=12, fontweight='semibold', labelpad=10)
ax.set_ylabel('Hypervolume (HV)', fontsize=12, fontweight='semibold', labelpad=10)

# Configure ticks
ax.set_xlim(-10, 510)
ax.set_ylim(0.40, 0.98)
ax.set_xticks(np.arange(0, 501, 50))
ax.set_yticks(np.arange(0.40, 1.00, 0.10))
ax.tick_params(axis='both', which='major', labelsize=10.5)

# Title (academic figure caption style)
plt.title("Figure 4: Hypervolume Convergence Curve across 500 Generations", 
          fontsize=13, pad=15, fontweight='bold')

# Elegant legend with white background and subtle grey border
legend = ax.legend(loc='lower right', frameon=True, facecolor='white', 
                   edgecolor='#d0d0d0', fontsize=11, shadow=False)
legend.get_frame().set_linewidth(0.8)

# Clean layout
plt.tight_layout()

# Save paths
workspace_img_path = 'visualize_convergence.png'
plt.savefig(workspace_img_path, dpi=300, bbox_inches='tight', facecolor='white')

# Save to the artifacts directory
artifacts_dir = r"C:\Users\bhagw\.gemini\antigravity-ide\brain\69f3e31f-50f3-488d-bb73-9ad30a523b08"
os.makedirs(artifacts_dir, exist_ok=True)
artifacts_img_path = os.path.join(artifacts_dir, 'visualize_convergence.png')
plt.savefig(artifacts_img_path, dpi=300, bbox_inches='tight', facecolor='white')

print(f"Plot successfully saved to artifacts directory: {artifacts_img_path}")
print(f"Plot successfully saved locally: {workspace_img_path}")
plt.close()
