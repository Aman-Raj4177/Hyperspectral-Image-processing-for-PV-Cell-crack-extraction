"""
Create a single-panel Component 1 image using the original style
(before smoothing/enhancement):
- no local smoothing
- no crack overlay
- nearest interpolation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = (r"c:\Users\amanraj.ASURITE\ASU Dropbox\Aman Raj\Aman Raj"
        r"\PV Reliability (HSI)\Analysis_02022026_Tripod_PV_field"
        r"\Crack Detection_0317")

COMP_CSV = os.path.join(BASE, "pls_component_scores_per_pixel.csv")
OUT_PNG = os.path.join(BASE, "component1_only.png")

comp = pd.read_csv(COMP_CSV)

x = comp["File X"].astype(int).to_numpy()
y = comp["File Y"].astype(int).to_numpy()
values = comp["PLS_Component_1"].to_numpy(float)

x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

h = y_max - y_min + 1
w = x_max - x_min + 1

grid = np.full((h, w), np.nan)
grid[y - y_min, x - x_min] = values

vmax = np.nanpercentile(np.abs(grid), 98)
if vmax == 0:
    vmax = 1.0

plt.figure(figsize=(7.2, 6.2), facecolor="#f8f8f8")
im = plt.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
plt.title("PLS Component 1", fontsize=13, fontweight="bold")
plt.xticks([])
plt.yticks([])
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label("Component 1 score")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
plt.close()

print(f"Saved: {OUT_PNG}")
