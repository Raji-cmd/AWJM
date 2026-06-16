import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === DATA ===
surface = pd.DataFrame({
    'Actual': [4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.5, 3.97, 4.63, 5.56, 3.62, 4.08, 4.75, 4.46,
               4.41, 3.42, 4.27, 4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.8],
    'Predicted': [4.65,3.23,4.43,5.01,3.87,5.05,4.34,4.65,3.97,3.57,4.04,4.52,5.56,3.65,4.16,4.71,4.43,4.34,3.43,4.25,
                 4.56,5.35,3.35,4.43,4.58,5.24,3.85]
})


mrr = pd.DataFrame({
    'Actual': [119.98, 126.36, 122.4, 111.57, 126.63, 97.63, 121.6, 109.81, 137.08, 117.28, 143.3, 125.7, 106.95,
               148.41, 102.02, 115.2, 122.4, 119.69, 145.92, 131.74, 132.71, 99.7, 140.98, 122.4, 104.26, 103.66,
               139.85],
    'Predicted': [118.15,128.31,120.37,114.03,128.86,98.40,119.45,112.40,133.38,122.63,143.81,122.22,108.80,150.03,103.46,
                 110.29,120.37,117.62,146.87,125.88,126.50,100.84,140.92,120.37,106.65,105.56,141.33]
})
kerf = pd.DataFrame({
    'Actual': [2.08, 1.44, 1.98, 2.25, 1.75, 2.25, 1.95, 2.09, 1.83, 1.59, 1.86, 2.04, 2.57, 1.64, 1.86, 2.15, 1.98,
               1.96, 1.52, 1.93, 2.05, 2.43, 1.48, 1.98, 2.08, 2.34, 1.73],
    'Predicted': [2.05,1.47,1.94,2.27,1.77,2.27,1.90,2.10,1.74,1.63,1.88,1.99,2.56,1.66,1.91,2.11,1.94,1.91,1.55,1.89,
                 1.99,2.44,1.53,1.94,2.10,2.35,1.76]
})

datasets = {
    "Surface Roughness": surface,
    "Material Removal Rate": mrr,
    "Kerf Angle": kerf
}

# === PLOT SEPARATE HEATMAPS WITHOUT ANNOTATION ===
for name, df in datasets.items():
    plt.figure(figsize=(12, 4))

    # Prepare 2xN matrix: row0=Actual, row1=Predicted
    heat_matrix = np.vstack([df['Actual'].values, df['Predicted'].values])

    im = plt.imshow(heat_matrix, aspect='auto', cmap='viridis', interpolation='nearest')

    # Labels
    plt.yticks([0, 1], ['Actual', 'Predicted'], fontweight='bold')
    plt.xlabel('Sample Index', fontweight='bold')
    plt.title(f'{name}: Actual vs Predicted Heat Map', fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Value', fontweight='bold')

    plt.tight_layout()
    plt.show()
