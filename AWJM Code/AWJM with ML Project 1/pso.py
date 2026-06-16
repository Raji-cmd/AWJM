import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

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
                                      104.26, 103.66, 139.85],
    'kerf_angle_deg': [2.08, 1.44, 1.98, 2.25, 1.75, 2.25, 1.95, 2.09, 1.83, 1.59,
                       1.86, 2.04, 2.57, 1.64, 1.86, 2.15, 1.98, 1.96, 1.52, 1.93,
                       2.05, 2.43, 1.48, 1.98, 2.08, 2.34, 1.73]
}

df = pd.DataFrame(data)

# ------------------- Plot 1: Surface Roughness -------------------
plt.figure(figsize=(8,6))
scatter = plt.scatter(df['pressure_MPa'], df['standoff_distance_mm'],
                      c=df['surface_roughness_um'], cmap='viridis',
                      s=100, alpha=0.8, edgecolors='black', linewidth=1.5)
plt.xlabel('Pressure (MPa)', fontsize=12, fontweight='bold')
plt.ylabel('Standoff Distance (mm)', fontsize=12, fontweight='bold')
plt.title('Surface Roughness, Ra (μm)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.3)
cbar = plt.colorbar(scatter)
cbar.set_label('Ra (μm)', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()

# ------------------- Plot 2: Material Removal Rate -------------------
plt.figure(figsize=(8,6))
scatter = plt.scatter(df['traverse_speed_mm_min'], df['mass_flow_rate_kg_min'],
                      c=df['material_removal_rate_mm3_min'], cmap='plasma',
                      s=100, alpha=0.8, edgecolors='black', linewidth=1.5)
plt.xlabel('Traverse Speed (mm/min)', fontsize=12, fontweight='bold')
plt.ylabel('Mass Flow Rate (kg/min)', fontsize=12, fontweight='bold')
plt.title('Material Removal Rate (mm³/min)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.3)
cbar = plt.colorbar(scatter)
cbar.set_label('MRR (mm³/min)', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()

# ------------------- Plot 3: Kerf Angle -------------------
plt.figure(figsize=(8,6))
scatter = plt.scatter(df['pressure_MPa'], df['traverse_speed_mm_min'],
                      c=df['kerf_angle_deg'], cmap='cool',
                      s=100, alpha=0.8, edgecolors='black', linewidth=1.5)
plt.xlabel('Pressure (MPa)', fontsize=12, fontweight='bold')
plt.ylabel('Traverse Speed (mm/min)', fontsize=12, fontweight='bold')
plt.title('Kerf Angle (°)', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.3)
cbar = plt.colorbar(scatter)
cbar.set_label('Kerf Angle (°)', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()
