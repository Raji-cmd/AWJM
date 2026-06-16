import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# ------------------- NEW AWJM DATA -------------------
data = {
    'pressure_MPa': [180, 240, 200, 160, 280, 150, 220, 200, 260, 300,
                     190, 250, 210, 170, 230, 240, 260, 180, 280, 220,
                     200, 160, 270, 210, 180, 230, 250],
    'standoff_distance_mm': [1.5, 2, 2.5, 1, 3, 2, 2.5, 1.5, 3, 1,
                             2, 2.5, 1, 3, 1.5, 2, 2, 1, 2.5, 2,
                             1, 2, 3, 1.5, 2, 2.5, 1],
    'traverse_speed_mm_min': [70, 90, 80, 60, 100, 75, 85, 95, 65, 90,
                              80, 70, 85, 60, 95, 100, 75, 80, 90, 85,
                              70, 95, 100, 60, 80, 85, 75],
    'mass_flow_rate_kg_min': [0.35, 0.5, 0.45, 0.55, 0.4, 0.45, 0.5, 0.35, 0.55, 0.4,
                              0.45, 0.35, 0.5, 0.55, 0.4, 0.5, 0.45, 0.35, 0.55, 0.4,
                              0.5, 0.55, 0.35, 0.45, 0.4, 0.5, 0.35],
    'surface_roughness_um': [4.20, 3.50, 4.10, 4.80, 3.75, 5.00, 4.35, 4.60, 3.90, 3.45,
                             4.05, 4.50, 5.10, 3.60, 4.25, 4.70, 4.15, 4.40, 3.55, 4.30,
                             4.55, 5.20, 3.35, 4.00, 4.50, 5.00, 3.80],
    'material_removal_rate_mm3_min': [115, 125, 120, 110, 130, 100, 122, 118, 135, 140,
                                      119, 127, 138, 105, 124, 132, 128, 121, 137, 125,
                                      129, 142, 108, 118, 126, 135, 104],
    'kerf_deg': [2.08, 1.44, 1.98, 2.25, 1.75, 2.25, 1.95, 2.09, 1.83, 1.59,
                 1.86, 2.04, 2.57, 1.64, 1.86, 2.15, 1.98, 1.96, 1.52, 1.93,
                 2.05, 2.43, 1.48, 1.98, 2.08, 2.34, 1.73]
}

# Create DataFrame
df = pd.DataFrame(data)

# Extract outputs
ra = df['surface_roughness_um'].values
mrr = df['material_removal_rate_mm3_min'].values
kerf = df['kerf_deg'].values

# ------------------- Identify Pareto-optimal solutions -------------------
def identify_pareto_optimal(scores):
    is_efficient = np.ones(scores.shape[0], dtype=bool)
    for i, c in enumerate(scores):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(scores[is_efficient] < c, axis=1) | \
                                        np.any(scores[is_efficient] > c, axis=1)
            is_efficient[i] = True
    return is_efficient

# Include kerf in the Pareto criteria (Min Ra, Max MRR, Min Kerf)
scores = np.column_stack([ra, -mrr, kerf])
pareto_mask = identify_pareto_optimal(scores)
pareto_points = np.column_stack([ra[pareto_mask], mrr[pareto_mask], kerf[pareto_mask]])
pareto_points = pareto_points[pareto_points[:, 1].argsort()]

# ------------------- Plot Pareto Frontier -------------------
plt.figure(figsize=(10, 8))
plt.scatter(ra, mrr, alpha=0.6, color='gray', s=50, label='Experimental Points', zorder=1)
plt.scatter(pareto_points[:,0], pareto_points[:,1], color='blue', s=80,
            label='Pareto-optimal Solutions', zorder=3, edgecolors='black', linewidth=1)
plt.plot(pareto_points[:,0], pareto_points[:,1], 'b--', alpha=0.7, linewidth=1.5, zorder=2)

# Key points
best_quality_idx = np.argmin(ra)
best_productivity_idx = np.argmax(mrr)
balanced_idx = np.argmin(np.sqrt((ra - ra.min())**2 + (mrr - mrr.max())**2))
key_points = [
    (ra[best_quality_idx], mrr[best_quality_idx], "Best Quality\n(Ra = {:.2f})".format(ra[best_quality_idx]), 's', 'red'),
    (ra[best_productivity_idx], mrr[best_productivity_idx], "Best Productivity\n(MRR = {:.2f})".format(mrr[best_productivity_idx]), '^', 'green'),
    (ra[balanced_idx], mrr[balanced_idx], "Balanced\n(Ra = {:.2f}, MRR = {:.2f})".format(ra[balanced_idx], mrr[balanced_idx]), 'D', 'purple')
]

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

# Feasible region (Convex Hull)
if len(pareto_points) >= 3:
    hull = ConvexHull(pareto_points[:, :2])  # only Ra vs MRR for plot
    plt.fill(pareto_points[hull.vertices,0], pareto_points[hull.vertices,1],
             'blue', alpha=0.1, label='Feasible Region')

plt.xlabel('Surface Roughness, Ra (μm)', fontsize=12, fontweight='bold')
plt.ylabel('Material Removal Rate (mm³/min)', fontsize=12, fontweight='bold')
plt.title('Multi-objective Pareto Frontier: Surface Quality vs Productivity',
          fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper left')
plt.xlim(ra.min()*0.95, ra.max()*1.05)
plt.ylim(mrr.min()*0.95, mrr.max()*1.05)
plt.tight_layout()
plt.show()

# ------------------- Summary Table -------------------
best_quality = df.iloc[best_quality_idx]
best_productivity = df.iloc[best_productivity_idx]
balanced = df.iloc[balanced_idx]

summary_table = pd.DataFrame({
    'Parameter': [
        'Pressure (MPa)', 'Standoff Distance (mm)', 'Traverse Speed (mm/min)',
        'Mass Flow Rate (kg/min)', 'Expected Ra (μm)', 'Expected MRR (mm³/min)', 'Kerf (deg)'
    ],
    'Best Quality (Min Ra)': [
        best_quality['pressure_MPa'], best_quality['standoff_distance_mm'], best_quality['traverse_speed_mm_min'],
        best_quality['mass_flow_rate_kg_min'], best_quality['surface_roughness_um'], best_quality['material_removal_rate_mm3_min'], best_quality['kerf_deg']
    ],
    'Best Productivity (Max MRR)': [
        best_productivity['pressure_MPa'], best_productivity['standoff_distance_mm'], best_productivity['traverse_speed_mm_min'],
        best_productivity['mass_flow_rate_kg_min'], best_productivity['surface_roughness_um'], best_productivity['material_removal_rate_mm3_min'], best_productivity['kerf_deg']
    ],
    'Balanced': [
        balanced['pressure_MPa'], balanced['standoff_distance_mm'], balanced['traverse_speed_mm_min'],
        balanced['mass_flow_rate_kg_min'], balanced['surface_roughness_um'], balanced['material_removal_rate_mm3_min'], balanced['kerf_deg']
    ]
})

print("="*80)
print("Table: Optimal Process Parameter Settings for Different Objectives (Including Kerf)")
print("="*80)
print(summary_table.to_string(index=False))
from mpl_toolkits.mplot3d import Axes3D

# ------------------- 3D Pareto Frontier Plot -------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# All points
ax.scatter(ra, mrr, kerf, color='gray', alpha=0.6, s=50, label='Experimental Points')

# Pareto-optimal points
ax.scatter(pareto_points[:,0], pareto_points[:,1], pareto_points[:,2],
           color='blue', s=80, edgecolors='black', label='Pareto-optimal Solutions')

ax.set_xlabel('Surface Roughness, Ra (μm)', fontsize=12, fontweight='bold')
ax.set_ylabel('Material Removal Rate (mm³/min)', fontsize=12, fontweight='bold')
ax.set_zlabel('Kerf (deg)', fontsize=12, fontweight='bold')
ax.set_title('Pareto Frontier: Ra vs MRR vs Kerf', fontsize=14, fontweight='bold', pad=20)

ax.legend()
plt.tight_layout()
plt.show()

