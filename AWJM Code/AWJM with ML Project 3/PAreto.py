import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# ------------------- NEW AWJM DATA (added kerf) -------------------
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
    'surface_roughness_um': [4.65, 3.40, 4.42, 4.91, 3.93, 4.96, 4.32, 4.63, 3.95, 3.67,
                             4.12, 4.51, 5.45, 3.69, 4.27, 4.71, 4.42, 4.32, 3.49, 4.24,
                             4.52, 5.24, 3.43, 4.42, 4.57, 5.18, 3.91],
    'Kerf_Angle_deg': [2.06, 1.59, 1.96, 2.17, 1.81, 2.21, 1.92, 2.07, 1.75, 1.69,
                       1.89, 2.00, 2.43, 1.68, 1.96, 2.10, 1.96, 1.93, 1.59, 1.90,
                       2.00, 2.34, 1.59, 1.96, 2.07, 2.29, 1.78]
}

# Create DataFrame
df = pd.DataFrame(data)

# Extract outputs
ra = df['surface_roughness_um'].values
mrr = df['material_removal_rate_mm3_min'].values
kerf = df['kerf_angle_deg'].values

# ------------------- Identify Pareto-optimal solutions -------------------
def identify_pareto_optimal(scores):
    is_efficient = np.ones(scores.shape[0], dtype=bool)
    for i, c in enumerate(scores):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(scores[is_efficient] < c, axis=1) | \
                                        np.any(scores[is_efficient] > c, axis=1)
            is_efficient[i] = True
    return is_efficient

# Objective: minimize Ra, minimize Kerf, maximize MRR
scores = np.column_stack([ra, -mrr, kerf])
pareto_mask = identify_pareto_optimal(scores)
pareto_points = np.column_stack([ra[pareto_mask], mrr[pareto_mask]])
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
best_kerf_idx = np.argmin(kerf)
balanced_idx = np.argmin(np.sqrt((ra - ra.min())**2 + (mrr - mrr.max())**2 + (kerf - kerf.min())**2))

key_points = [
    (ra[best_quality_idx], mrr[best_quality_idx], "Best Quality\n(Min Ra = {:.2f})".format(ra[best_quality_idx]), 's', 'red'),
    (ra[best_productivity_idx], mrr[best_productivity_idx], "Best Productivity\n(Max MRR = {:.2f})".format(mrr[best_productivity_idx]), '^', 'green'),
    (ra[best_kerf_idx], mrr[best_kerf_idx], "Best Kerf\n(Min Kerf = {:.2f})".format(kerf[best_kerf_idx]), 'v', 'orange'),
    (ra[balanced_idx], mrr[balanced_idx], "Balanced\n(Ra={:.2f}, MRR={:.2f}, Kerf={:.2f})".format(ra[balanced_idx], mrr[balanced_idx], kerf[balanced_idx]), 'D', 'purple')
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

# Feasible region
if len(pareto_points) >= 3:
    hull = ConvexHull(pareto_points)
    plt.fill(pareto_points[hull.vertices,0], pareto_points[hull.vertices,1],
             'blue', alpha=0.1, label='Feasible Region')

plt.xlabel('Surface Roughness, Ra (μm)', fontsize=12, fontweight='bold')
plt.ylabel('Material Removal Rate (mm³/min)', fontsize=12, fontweight='bold')
plt.title('Multi-objective Pareto Frontier: Ra vs MRR (with Kerf Considered)',
          fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(
    loc='upper left',
    fontsize=9,          # smaller text
    framealpha=0.7,      # slightly transparent
    markerscale=0.8,     # smaller legend markers
    handlelength=1.5,    # shorter legend lines
    borderpad=0.4,       # tighter padding inside box
    labelspacing=0.3     # less spacing between entries
)

plt.xlim(ra.min()*0.95, ra.max()*1.05)
plt.ylim(mrr.min()*0.95, mrr.max()*1.05)
plt.tight_layout()
plt.show()

# ------------------- Summary Table -------------------
best_quality = df.iloc[best_quality_idx]
best_productivity = df.iloc[best_productivity_idx]
best_kerf = df.iloc[best_kerf_idx]
balanced = df.iloc[balanced_idx]

summary_table = pd.DataFrame({
    'Parameter': [
        'Pressure (MPa)', 'Standoff Distance (mm)', 'Traverse Speed (mm/min)',
        'Mass Flow Rate (kg/min)', 'Expected Ra (μm)',
        'Expected MRR (mm³/min)', 'Expected Kerf (°)'
    ],
    'Best Quality (Min Ra)': [
        best_quality['pressure_MPa'], best_quality['standoff_distance_mm'], best_quality['traverse_speed_mm_min'],
        best_quality['mass_flow_rate_kg_min'], best_quality['surface_roughness_um'],
        best_quality['material_removal_rate_mm3_min'], best_quality['kerf_angle_deg']
    ],
    'Best Productivity (Max MRR)': [
        best_productivity['pressure_MPa'], best_productivity['standoff_distance_mm'], best_productivity['traverse_speed_mm_min'],
        best_productivity['mass_flow_rate_kg_min'], best_productivity['surface_roughness_um'],
        best_productivity['material_removal_rate_mm3_min'], best_productivity['kerf_angle_deg']
    ],
    'Best Kerf (Min Kerf)': [
        best_kerf['pressure_MPa'], best_kerf['standoff_distance_mm'], best_kerf['traverse_speed_mm_min'],
        best_kerf['mass_flow_rate_kg_min'], best_kerf['surface_roughness_um'],
        best_kerf['material_removal_rate_mm3_min'], best_kerf['kerf_angle_deg']
    ],
    'Balanced': [
        balanced['pressure_MPa'], balanced['standoff_distance_mm'], balanced['traverse_speed_mm_min'],
        balanced['mass_flow_rate_kg_min'], balanced['surface_roughness_um'],
        balanced['material_removal_rate_mm3_min'], balanced['kerf_angle_deg']
    ]
})

print("="*80)
print("Table: Optimal Process Parameter Settings for Different Objectives (Including Kerf)")
print("="*80)
print(summary_table.to_string(index=False))
# ===== Save summary table to Excel =====
summary_table.to_excel("AWJM_Pareto_Optimal_Results.xlsx", index=False)
print("\n✅ Summary table successfully saved as 'AWJM_Pareto_Optimal_Results.xlsx'")
