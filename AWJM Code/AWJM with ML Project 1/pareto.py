import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import matplotlib.patches as mpatches

# ------------------- AWJM Data -------------------
data = {
    'pressure_MPa': [220, 260, 220, 140, 300, 140, 220, 140, 280, 300,
                     300, 220, 140, 300, 140, 180, 220, 220, 300, 220,
                     220, 140, 300, 220, 140, 140, 300],
    'standoff_distance_mm': [2, 1, 2, 3, 1, 1, 2, 3, 2, 1,
                             3, 2, 3, 3, 1, 2, 2, 1.5, 3, 2,
                             2.5, 1, 1, 2, 1, 3, 3],
    'traverse_speed_mm_min': [80, 64, 80, 96, 96, 64, 72, 64, 80, 64,
                              96, 88, 96, 96, 64, 80, 80, 80, 64, 80,
                              80, 96, 96, 80, 96, 64, 64],
    'mass_flow_rate_kg_min': [0.4, 0.55, 0.45, 0.55, 0.35, 0.35, 0.45, 0.55, 0.45, 0.35,
                              0.35, 0.45, 0.35, 0.55, 0.55, 0.45, 0.45, 0.45, 0.55, 0.5,
                              0.45, 0.35, 0.55, 0.45, 0.55, 0.35, 0.35],
    'surface_roughness_um': [4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.5,
                             3.97, 4.63, 5.56, 3.62, 4.08, 4.75, 4.46, 4.41, 3.42, 4.27,
                             4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.8],
    'material_removal_rate_mm3_min': [119.98, 126.36, 122.4, 111.57, 126.63, 97.63,
                                      121.6, 109.81, 137.08, 117.28, 143.3, 125.7,
                                      106.95, 148.41, 102.02, 115.2, 122.4, 119.69,
                                      145.92, 131.74, 132.71, 99.7, 140.98, 122.4,
                                      104.26, 103.66, 139.85]
}

# Create DataFrame
df = pd.DataFrame(data)

# Extract the output responses
ra = df['surface_roughness_um'].values
mrr = df['material_removal_rate_mm3_min'].values

# ------------------- Identify Pareto-optimal solutions -------------------
def identify_pareto_optimal(scores):
    """Identify Pareto-optimal solutions (minimize Ra, maximize MRR)"""
    is_efficient = np.ones(scores.shape[0], dtype=bool)
    for i, c in enumerate(scores):
        if is_efficient[i]:
            # For minimization of Ra and maximization of MRR
            is_efficient[is_efficient] = np.any(scores[is_efficient] < c, axis=1) | \
                                        np.any(scores[is_efficient] > c, axis=1)
            is_efficient[i] = True
    return is_efficient

# Create score array (we want to minimize Ra and maximize MRR)
scores = np.column_stack([ra, -mrr])  # Negative MRR for minimization
pareto_mask = identify_pareto_optimal(scores)
pareto_points = np.column_stack([ra[pareto_mask], mrr[pareto_mask]])

# Sort Pareto points by MRR for better visualization
pareto_points = pareto_points[pareto_points[:, 1].argsort()]

# ------------------- Create the Pareto Frontier Plot (Clean Version) -------------------
plt.figure(figsize=(10, 8))

# Plot all experimental points
plt.scatter(ra, mrr, alpha=0.6, color='gray', s=50, label='Experimental Points', zorder=1)

# Plot Pareto-optimal solutions
plt.scatter(pareto_points[:, 0], pareto_points[:, 1], color='blue', s=80,
            label='Pareto-optimal Solutions', zorder=3, edgecolors='black', linewidth=1)

# Connect Pareto points with lines
plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'b--', alpha=0.7, linewidth=1.5, zorder=2)

# Identify key points
best_quality_idx = np.argmin(ra)
best_productivity_idx = np.argmax(mrr)
balanced_idx = np.argmin(np.sqrt((ra - ra.min())**2 + (mrr - mrr.max())**2))

# Define key points for plotting
key_points = [
    (ra[best_quality_idx], mrr[best_quality_idx], "Best Quality\n(Ra = {:.2f})".format(ra[best_quality_idx]), 's', 'red'),
    (ra[best_productivity_idx], mrr[best_productivity_idx], "Best Productivity\n(MRR = {:.2f})".format(mrr[best_productivity_idx]), '^', 'green'),
    (ra[balanced_idx], mrr[balanced_idx], "Balanced\n(Ra = {:.2f}, MRR = {:.2f})".format(ra[balanced_idx], mrr[balanced_idx]), 'D', 'purple')
]

# Plot and annotate key points
for x, y, label, marker, color in key_points:
    plt.scatter(x, y, s=150, marker=marker, color=color, edgecolors='black', linewidth=2, zorder=4)
    offset = (70, -30) if "Balanced" in label else (10, 10)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=offset,
                 textcoords='offset points',
                 fontsize=10,
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Shaded feasible region (Convex Hull)
if len(pareto_points) >= 3:
    hull = ConvexHull(pareto_points)
    plt.fill(pareto_points[hull.vertices, 0], pareto_points[hull.vertices, 1],
             'blue', alpha=0.1, label='Feasible Region')

# Customize plot
plt.xlabel('Surface Roughness, Ra (μm)', fontsize=12, fontweight='bold')
plt.ylabel('Material Removal Rate (mm³/min)', fontsize=12, fontweight='bold')
plt.title('Figure 10: Multi-objective Pareto Frontier for AWJM Process\nTrade-off between Surface Quality and Productivity',
          fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper right')

# Set axis limits
plt.xlim(ra.min() * 0.95, ra.max() * 1.05)
plt.ylim(mrr.min() * 0.95, mrr.max() * 1.05)

plt.tight_layout()
plt.show()


# Print summary of Pareto-optimal solutions
print("="*70)
print("PARETO-OPTIMAL SOLUTIONS SUMMARY")
print("="*70)
print(f"Number of Pareto-optimal solutions: {pareto_mask.sum()}")
print(f"Best Quality - Ra: {ra[best_quality_idx]:.2f} μm, MRR: {mrr[best_quality_idx]:.2f} mm³/min")
print(f"Best Productivity - Ra: {ra[best_productivity_idx]:.2f} μm, MRR: {mrr[best_productivity_idx]:.2f} mm³/min")
print(f"Balanced Operation - Ra: {ra[balanced_idx]:.2f} μm, MRR: {mrr[balanced_idx]:.2f} mm³/min")

# Display the table
print("Table 3: Optimal Process Parameter Settings for Different Objectives\n")
print(df.to_string(index=False))

# ====================== Summary Table ======================
best_quality = df.iloc[best_quality_idx]
best_productivity = df.iloc[best_productivity_idx]
balanced = df.iloc[balanced_idx]

summary_table = pd.DataFrame({
    'Parameter': [
        'Pressure (MPa)',
        'Standoff Distance (mm)',
        'Traverse Speed (mm/min)',
        'Mass Flow Rate (kg/min)',
        'Expected Ra (μm)',
        'Expected MRR (mm³/min)'
    ],
    'Best Quality (Min Ra)': [
        best_quality['pressure_MPa'],
        best_quality['standoff_distance_mm'],
        best_quality['traverse_speed_mm_min'],
        best_quality['mass_flow_rate_kg_min'],
        best_quality['surface_roughness_um'],
        best_quality['material_removal_rate_mm3_min']
    ],
    'Best Productivity (Max MRR)': [
        best_productivity['pressure_MPa'],
        best_productivity['standoff_distance_mm'],
        best_productivity['traverse_speed_mm_min'],
        best_productivity['mass_flow_rate_kg_min'],
        best_productivity['surface_roughness_um'],
        best_productivity['material_removal_rate_mm3_min']
    ],
    'Balanced': [
        balanced['pressure_MPa'],
        balanced['standoff_distance_mm'],
        balanced['traverse_speed_mm_min'],
        balanced['mass_flow_rate_kg_min'],
        balanced['surface_roughness_um'],
        balanced['material_removal_rate_mm3_min']
    ]
})

print("\n" + "="*70)
print("Table 3: Optimal Process Parameter Settings for Different Objectives")
print("="*70)
print(summary_table.to_string(index=False))