"""
Savitzky-Golay spectral smoothing applied independently to every pixel
in VNIR_1x1_cellcoordinate.csv.

Spectral axis : 200 bands (B1–B200), representing ~400–1000 nm
window_length : 11   (must be odd and > polyorder)
polyorder     : 3
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# ── Parameters ─────────────────────────────────────────────────────────────
INPUT_CSV  = r"c:\Users\amanraj.ASURITE\ASU Dropbox\Aman Raj\Aman Raj\PV Reliability (HSI)\Analysis_02022026_Tripod_PV_field\Crack Detection_0317\VNIR_1x1_cellcoordinate.csv"
OUTPUT_CSV = r"c:\Users\amanraj.ASURITE\ASU Dropbox\Aman Raj\Aman Raj\PV Reliability (HSI)\Analysis_02022026_Tripod_PV_field\Crack Detection_0317\VNIR_1x1_SavGol_smoothed.csv"

WINDOW_LENGTH = 11
POLYORDER     = 3

# ── Validate parameters ─────────────────────────────────────────────────────
assert WINDOW_LENGTH % 2 == 1,            "window_length must be odd"
assert WINDOW_LENGTH > POLYORDER,         "window_length must be greater than polyorder"

# ── Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()          # strip any accidental whitespace

coord_cols   = ["File X", "File Y"]
band_cols    = [c for c in df.columns if c.startswith("B")]

print(f"Loaded  : {len(df)} pixels  |  {len(band_cols)} spectral bands")
print(f"SG filter: window_length={WINDOW_LENGTH}, polyorder={POLYORDER}")

# ── Extract spectral matrix (pixels × bands) ────────────────────────────────
spectra = df[band_cols].values.astype(float)   # shape: (n_pixels, 200)

# ── Apply SG filter along band axis (axis=1) for every pixel ────────────────
smoothed = savgol_filter(spectra, window_length=WINDOW_LENGTH,
                         polyorder=POLYORDER, axis=1)

# ── Build output DataFrame ─────────────────────────────────────────────────
df_out = df[coord_cols].copy()
df_smooth_bands = pd.DataFrame(smoothed, columns=band_cols, index=df.index)
df_out = pd.concat([df_out, df_smooth_bands], axis=1)

# ── Save ───────────────────────────────────────────────────────────────────
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"Saved   : {OUTPUT_CSV}")

# ── Quick quality check ────────────────────────────────────────────────────
pixel_idx   = 0                         # first pixel as example
orig_pixel  = spectra[pixel_idx]
smooth_pixel= smoothed[pixel_idx]
residuals   = orig_pixel - smooth_pixel

print("\n── Sample pixel 0 (File X={}, File Y={}) ──".format(
    df["File X"].iloc[0], df["File Y"].iloc[0]))
print(f"  Band range raw    : {orig_pixel.min():.4f} – {orig_pixel.max():.4f}")
print(f"  Band range smooth : {smooth_pixel.min():.4f} – {smooth_pixel.max():.4f}")
print(f"  Max residual      : {np.abs(residuals).max():.4f}")
print(f"  Mean residual     : {np.abs(residuals).mean():.4f}")
print(f"  Std  residual     : {residuals.std():.4f}")

print("\nDone.")
