"""
Spatial visualization of PLS components for each pixel.

Uses trained tuned PLS-DA model and smoothed spectra to compute latent
X-scores (PLS components) per pixel, then maps each component spatially.

Outputs:
- pls_component_scores_per_pixel.csv
- pls_component_spatial_maps.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

BASE = (r"c:\Users\amanraj.ASURITE\ASU Dropbox\Aman Raj\Aman Raj"
        r"\PV Reliability (HSI)\Analysis_02022026_Tripod_PV_field"
        r"\Crack Detection_0317")

INPUT_CSV = os.path.join(BASE, "VNIR_1x1_SavGol_smoothed.csv")
MODEL_FILE = os.path.join(BASE, "plsda_tuned_model.joblib")
OUT_CSV = os.path.join(BASE, "pls_component_scores_per_pixel.csv")
OUT_PNG = os.path.join(BASE, "pls_component_spatial_maps.png")

# Load model/data
model = joblib.load(MODEL_FILE)
df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()

all_band_cols = model["band_cols_all"]
selected_idx = np.array(model["selected_idx"], dtype=int)
selected_bands = [all_band_cols[i] for i in selected_idx]

X = df[selected_bands].to_numpy(float)
X_scaled = model["scaler"].transform(X)

# PLS latent scores for each sample (pixel)
# shape: (n_pixels, n_components)
scores = model["pls_model"].transform(X_scaled)

n_pixels, n_comp = scores.shape

# Save per-pixel component scores
out_df = df[["File X", "File Y"]].copy()
for i in range(n_comp):
    out_df[f"PLS_Component_{i+1}"] = scores[:, i]
out_df.to_csv(OUT_CSV, index=False)

# Build spatial geometry
x = df["File X"].astype(int).to_numpy()
y = df["File Y"].astype(int).to_numpy()

x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

h = y_max - y_min + 1
w = x_max - x_min + 1

# Plot component maps
ncols = 3
nrows = int(np.ceil(n_comp / ncols))
fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(5.5 * ncols, 4.8 * nrows),
    facecolor="#f8f8f8"
)
axes = np.atleast_1d(axes).ravel()

for i in range(n_comp):
    grid = np.full((h, w), np.nan)
    grid[y - y_min, x - x_min] = scores[:, i]

    vmax = np.nanpercentile(np.abs(grid), 98)
    if vmax == 0:
        vmax = 1.0

    ax = axes[i]
    im = ax.imshow(
        grid,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest"
    )
    ax.set_title(f"PLS Component {i+1}")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Score")

for i in range(n_comp, len(axes)):
    axes[i].axis("off")

fig.suptitle("Spatial Maps of PLS Components (Per Pixel)", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT_PNG, dpi=180)
plt.close()

print("Saved:")
print(f"  {OUT_CSV}")
print(f"  {OUT_PNG}")
print(f"Components visualized: {n_comp}")
print(f"Pixels: {n_pixels}")
