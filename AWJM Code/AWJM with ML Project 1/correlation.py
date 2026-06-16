import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Input Parameters ===
inputs = {
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
                              0.45, 0.35, 0.55, 0.45, 0.55, 0.35, 0.35]
}

# === Output Parameters ===
outputs = {
    'Surface_Roughness': [4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.5,
                             3.97, 4.63, 5.56, 3.62, 4.08, 4.75, 4.46, 4.41, 3.42, 4.27,
                             4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.8],
    'Material_Removal_Rate': [119.98,126.36,122.4,111.57,126.63,97.63,121.6,109.81,137.08,117.28,
                                      143.3,125.7,106.95,148.41,102.02,115.2,122.4,119.69,145.92,131.74,
                                      132.71,99.7,140.98,122.4,104.26,103.66,139.85],
    'Kerf_Angle': [2.08,1.44,1.98,2.25,1.75,2.25,1.95,2.09,1.83,1.59,
                       1.86,2.04,2.57,1.64,1.86,2.15,1.98,1.96,1.52,1.93,
                       2.05,2.43,1.48,1.98,2.08,2.34,1.73]
}

# === Combine into one DataFrame ===
df = pd.DataFrame({**inputs, **outputs})

# === Compute correlation matrix ===
corr_matrix = df.corr()

# === Plot heatmap ===
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

plt.title("Correlation Matrix ", fontsize=16, fontweight='bold', pad=15)
plt.xticks(rotation=90, ha='right', fontsize=11, fontweight='bold')
plt.yticks(rotation=0, fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()
