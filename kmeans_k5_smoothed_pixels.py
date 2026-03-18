"""
Run k-means (k=5) on Savitzky-Golay smoothed VNIR spectra and create labeled pixels.

Input : VNIR_1x1_SavGol_smoothed.csv
Output: VNIR_1x1_SavGol_kmeans_k5_labels.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE = (r"c:\Users\amanraj.ASURITE\ASU Dropbox\Aman Raj\Aman Raj"
        r"\PV Reliability (HSI)\Analysis_02022026_Tripod_PV_field"
        r"\Crack Detection_0317")

INPUT_CSV = os.path.join(BASE, "VNIR_1x1_SavGol_smoothed.csv")
OUTPUT_CSV = os.path.join(BASE, "VNIR_1x1_SavGol_kmeans_k5_labels.csv")

K = 5
RANDOM_STATE = 42

# Load smoothed reflectance data
_df = pd.read_csv(INPUT_CSV)
_df.columns = _df.columns.str.strip()

coord_cols = ["File X", "File Y"]
band_cols = [c for c in _df.columns if c.startswith("B")]

if len(band_cols) != 200:
    raise ValueError(f"Expected 200 spectral bands, found {len(band_cols)}")

X = _df[band_cols].to_numpy(dtype=float)

# Standardize features so all bands contribute comparably to Euclidean distance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means clustering
kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=20)
labels = kmeans.fit_predict(X_scaled)

# Save labeled pixels (coordinates + cluster label)
out_df = _df[coord_cols].copy()
out_df["Cluster_Label"] = labels
out_df.to_csv(OUTPUT_CSV, index=False)

# Print summary for quick validation
counts = pd.Series(labels).value_counts().sort_index()
print(f"Input pixels: {len(_df)}")
print(f"Bands used  : {len(band_cols)}")
print(f"k           : {K}")
print(f"Output file : {OUTPUT_CSV}")
print("\nCluster counts:")
for c, n in counts.items():
    print(f"  Cluster {c}: {n}")

# Optional clustering quality index (lower inertia is tighter clusters)
print(f"\nKMeans inertia: {kmeans.inertia_:.4f}")
