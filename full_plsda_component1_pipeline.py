"""
Full end-to-end workflow in a single Python script.

Pipeline:
1. Load raw VNIR spectra
2. Savitzky-Golay smoothing
3. K-means (k=5) pseudo-label generation
4. Tuned PLS-DA training with balancing + optional VIP selection
5. Apply trained PLS-DA to all pixels
6. Generate class-score maps, class map, crack map
7. Generate PLS component score table and component maps
8. Generate final Component 1 image

Primary final output:
- component1_only.png
"""

from __future__ import annotations

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

WINDOW_LENGTH = 11
POLYORDER = 3
K = 5
RANDOM_STATE = 42
TEST_SIZE = 0.25
N_SPLITS = 5
COMPONENT_GRID = list(range(2, 11))
VIP_THRESHOLD = 1.0


def parse_args() -> argparse.Namespace:
    default_input = os.path.join(SCRIPT_DIR, "VNIR_1x1_cellcoordinate.csv")
    parser = argparse.ArgumentParser(
        description="Full end-to-end PLS-DA Component 1 pipeline for hyperspectral crack analysis."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=default_input,
        help="Path to raw hyperspectral CSV input file. Default: VNIR_1x1_cellcoordinate.csv in the script folder.",
    )
    return parser.parse_args()


def extract_size_token(input_csv: str) -> tuple[str, str]:
    stem = os.path.splitext(os.path.basename(input_csv))[0]
    match = re.search(r"_(\d+)x(\d+)_", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)

    match = re.search(r"(\d+)x(\d+)", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)

    raise ValueError(
        "Could not extract the size token from the input filename. "
        "Expected a pattern like '..._1x1_...' or '...1x1...'."
    )


def build_output_paths(input_csv: str) -> dict[str, str]:
    input_dir = os.path.dirname(os.path.abspath(input_csv))
    left, right = extract_size_token(input_csv)
    output_dir = os.path.join(input_dir, f"Output_{left} x {right}")
    os.makedirs(output_dir, exist_ok=True)

    return {
        "RAW_CSV": os.path.abspath(input_csv),
        "OUTPUT_DIR": output_dir,
        "SMOOTH_CSV": os.path.join(output_dir, "VNIR_SavGol_smoothed.csv"),
        "KMEANS_LABEL_CSV": os.path.join(output_dir, "VNIR_SavGol_kmeans_k5_labels.csv"),
        "KMEANS_SUMMARY_CSV": os.path.join(output_dir, "kmeans_k5_cluster_summary.csv"),
        "KMEANS_MEAN_SPECTRA_PNG": os.path.join(output_dir, "kmeans_k5_cluster_mean_spectra.png"),
        "KMEANS_SPATIAL_MAPS_PNG": os.path.join(output_dir, "kmeans_k5_spatial_maps.png"),
        "PLS_MODEL_FILE": os.path.join(output_dir, "plsda_tuned_model.joblib"),
        "PLS_TEST_PRED_CSV": os.path.join(output_dir, "plsda_tuned_test_predictions.csv"),
        "PLS_CM_PNG": os.path.join(output_dir, "plsda_tuned_confusion_matrix.png"),
        "PLS_METRICS_TXT": os.path.join(output_dir, "plsda_tuned_metrics.txt"),
        "PLS_CV_RESULTS_CSV": os.path.join(output_dir, "plsda_cv_results.csv"),
        "PLS_VIP_CSV": os.path.join(output_dir, "plsda_vip_scores.csv"),
        "MODULE_PRED_CSV": os.path.join(output_dir, "plsda_module_predictions.csv"),
        "CLASS_SCORE_MAPS_PNG": os.path.join(output_dir, "plsda_class_score_maps.png"),
        "FINAL_CLASS_MAP_PNG": os.path.join(output_dir, "plsda_final_class_map.png"),
        "CRACK_MAP_PNG": os.path.join(output_dir, "plsda_crack_map.png"),
        "CRACK_SUMMARY_TXT": os.path.join(output_dir, "plsda_crack_extent_summary.txt"),
        "PLS_COMPONENT_CSV": os.path.join(output_dir, "pls_component_scores_per_pixel.csv"),
        "PLS_COMPONENT_MAPS_PNG": os.path.join(output_dir, "pls_component_spatial_maps.png"),
        "COMPONENT1_ONLY_PNG": os.path.join(output_dir, "component1_only.png"),
    }


def ensure_valid_savgol_params(window_length: int, polyorder: int) -> None:
    if window_length % 2 != 1:
        raise ValueError("window_length must be odd")
    if window_length <= polyorder:
        raise ValueError("window_length must be greater than polyorder")



def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()



def softmax(a: np.ndarray) -> np.ndarray:
    a = a - np.max(a, axis=1, keepdims=True)
    e = np.exp(a)
    return e / np.sum(e, axis=1, keepdims=True)



def build_xy_grid(x: np.ndarray, y: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, int, int, int, int]:
    x_min, x_max = int(x.min()), int(x.max())
    y_min, y_max = int(y.min()), int(y.max())
    h = y_max - y_min + 1
    w = x_max - x_min + 1
    grid = np.full((h, w), np.nan)
    grid[y - y_min, x - x_min] = values
    return grid, x_min, x_max, y_min, y_max



def oversample_to_max_class(X: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    idx_all = []

    for cls in classes:
        idx = np.where(y == cls)[0]
        if len(idx) < max_count:
            extra = rng.choice(idx, size=max_count - len(idx), replace=True)
            idx = np.concatenate([idx, extra])
        idx_all.append(idx)

    idx_bal = np.concatenate(idx_all)
    rng.shuffle(idx_bal)
    return X[idx_bal], y[idx_bal]



def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    return np.eye(n_classes)[y]



def compute_vip(pls: PLSRegression) -> np.ndarray:
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_

    p = w.shape[0]
    h = w.shape[1]
    s = np.zeros(h)

    for a in range(h):
        s[a] = np.sum(t[:, a] ** 2) * np.sum(q[:, a] ** 2)

    total_s = np.sum(s)
    if total_s <= 0:
        return np.ones(p)

    vip = np.zeros(p)
    for j in range(p):
        num = 0.0
        for a in range(h):
            denom = np.sum(w[:, a] ** 2)
            if denom > 0:
                num += s[a] * ((w[j, a] ** 2) / denom)
        vip[j] = np.sqrt(p * num / total_s)
    return vip



def fit_predict_plsda(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, *, n_components: int, use_vip: bool, n_classes: int, seed: int) -> tuple[np.ndarray, dict]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    X_bal, y_bal = oversample_to_max_class(X_train_s, y_train, seed=seed)
    Y_bal = one_hot(y_bal, n_classes)

    pls = PLSRegression(n_components=n_components)
    pls.fit(X_bal, Y_bal)

    selected_idx = np.arange(X_train.shape[1])
    vip_scores = None

    if use_vip:
        vip_scores = compute_vip(pls)
        selected_idx = np.where(vip_scores >= VIP_THRESHOLD)[0]
        if len(selected_idx) < 10:
            selected_idx = np.argsort(vip_scores)[-30:]

        X_train_v = X_train[:, selected_idx]
        X_val_v = X_val[:, selected_idx]

        scaler = StandardScaler()
        X_train_vs = scaler.fit_transform(X_train_v)
        X_val_vs = scaler.transform(X_val_v)

        X_bal, y_bal = oversample_to_max_class(X_train_vs, y_train, seed=seed)
        Y_bal = one_hot(y_bal, n_classes)

        pls = PLSRegression(n_components=min(n_components, X_bal.shape[1]))
        pls.fit(X_bal, Y_bal)
        val_scores = pls.predict(X_val_vs)
    else:
        val_scores = pls.predict(X_val_s)

    y_pred = np.argmax(val_scores, axis=1)
    bundle = {
        "scaler": scaler,
        "pls": pls,
        "selected_idx": selected_idx,
        "use_vip": use_vip,
        "vip_scores": vip_scores,
        "n_components": n_components,
    }
    return y_pred, bundle



def stage_1_savgol(paths: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    ensure_valid_savgol_params(WINDOW_LENGTH, POLYORDER)
    df = pd.read_csv(paths["RAW_CSV"])
    df.columns = df.columns.str.strip()
    band_cols = [c for c in df.columns if c.startswith("B")]

    spectra = df[band_cols].to_numpy(float)
    smoothed = savgol_filter(spectra, window_length=WINDOW_LENGTH, polyorder=POLYORDER, axis=1)

    out = pd.concat(
        [df[["File X", "File Y"]].reset_index(drop=True), pd.DataFrame(smoothed, columns=band_cols)],
        axis=1,
    )
    out.to_csv(paths["SMOOTH_CSV"], index=False)
    return out, band_cols



def stage_2_kmeans(df_smooth: pd.DataFrame, band_cols: list[str], paths: dict[str, str]) -> pd.DataFrame:
    X = df_smooth[band_cols].to_numpy(float)
    X_scaled = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=20).fit_predict(X_scaled)

    out = df_smooth[["File X", "File Y"]].copy()
    out["Cluster_Label"] = labels
    out.to_csv(paths["KMEANS_LABEL_CSV"], index=False)

    merged = df_smooth.merge(out, on=["File X", "File Y"], how="inner")
    wavelengths = np.linspace(400, 1000, len(band_cols))
    clusters = sorted(np.unique(labels).tolist())

    # Mean spectra plot
    plt.figure(figsize=(12, 7), facecolor="#f9f9f9")
    rows = []
    for cls in clusters:
        arr = merged.loc[merged["Cluster_Label"] == cls, band_cols].to_numpy(float)
        mean_arr = arr.mean(axis=0)
        std_arr = arr.std(axis=0)
        plt.plot(wavelengths, mean_arr, lw=2.0, label=f"Cluster {cls}")
        plt.fill_between(wavelengths, mean_arr - std_arr, mean_arr + std_arr, alpha=0.15)

        idx_400 = np.argmin(np.abs(wavelengths - 400))
        idx_550 = np.argmin(np.abs(wavelengths - 550))
        idx_700 = np.argmin(np.abs(wavelengths - 700))
        idx_900 = np.argmin(np.abs(wavelengths - 900))
        rows.append({
            "Cluster_Label": cls,
            "Pixel_Count": int((labels == cls).sum()),
            "Mean_400nm": float(mean_arr[idx_400]),
            "Mean_550nm": float(mean_arr[idx_550]),
            "Mean_700nm": float(mean_arr[idx_700]),
            "Mean_900nm": float(mean_arr[idx_900]),
            "Slope_700_to_900_per_nm": float((mean_arr[idx_900] - mean_arr[idx_700]) / (wavelengths[idx_900] - wavelengths[idx_700])),
            "Mean_AllBands": float(mean_arr.mean()),
            "Std_AllBands": float(mean_arr.std()),
        })

    plt.title("K-Means (k=5): Mean Spectrum of Each Cluster", fontsize=13, fontweight="bold")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Smoothed Reflectance")
    plt.xlim(400, 1000)
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_fig(paths["KMEANS_MEAN_SPECTRA_PNG"])

    summary = pd.DataFrame(rows).sort_values("Cluster_Label")
    summary.to_csv(paths["KMEANS_SUMMARY_CSV"], index=False)

    # Spatial maps
    x = merged["File X"].astype(int).to_numpy()
    y = merged["File Y"].astype(int).to_numpy()
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    h = y_max - y_min + 1
    w = x_max - x_min + 1

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="#f9f9f9")
    axes = axes.ravel()
    for i, cls in enumerate(clusters):
        grid = np.full((h, w), np.nan)
        mask = labels == cls
        grid[y[mask] - y_min, x[mask] - x_min] = 1.0
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="white")
        axes[i].imshow(grid, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
        axes[i].set_title(f"Cluster {cls}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    label_grid = np.full((h, w), np.nan)
    label_grid[y - y_min, x - x_min] = labels
    im = axes[-1].imshow(label_grid, cmap="tab10", interpolation="nearest", vmin=min(clusters), vmax=max(clusters))
    axes[-1].set_title("All Clusters")
    axes[-1].set_xticks([])
    axes[-1].set_yticks([])
    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04).set_label("Cluster Label")
    fig.suptitle("K-Means (k=5): Spatial Map of Each Cluster", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(paths["KMEANS_SPATIAL_MAPS_PNG"], dpi=180)
    plt.close()

    return out



def stage_3_train_plsda(df_smooth: pd.DataFrame, df_labels: pd.DataFrame, band_cols: list[str], paths: dict[str, str]) -> dict:
    merged = df_smooth.merge(df_labels, on=["File X", "File Y"], how="inner")
    X = merged[band_cols].to_numpy(float)
    y_raw = merged["Cluster_Label"].to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = np.unique(y)
    class_names = le.inverse_transform(classes)
    n_classes = len(classes)

    target_clusters = [3, 4]
    idx_c3 = int(np.where(class_names == 3)[0][0])
    idx_c4 = int(np.where(class_names == 4)[0][0])

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        np.arange(len(y)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    rows = []
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for use_vip in [False, True]:
        mode = "VIP" if use_vip else "FULL"
        for n_comp in COMPONENT_GRID:
            fold_c34 = []
            fold_macro = []
            for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
                X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
                X_va, y_va = X_train[va_idx], y_train[va_idx]
                y_hat, _ = fit_predict_plsda(
                    X_tr,
                    y_tr,
                    X_va,
                    n_components=n_comp,
                    use_vip=use_vip,
                    n_classes=n_classes,
                    seed=RANDOM_STATE + fold_id,
                )
                recalls = recall_score(y_va, y_hat, labels=classes, average=None, zero_division=0)
                fold_c34.append((recalls[idx_c3] + recalls[idx_c4]) / 2.0)
                fold_macro.append(recall_score(y_va, y_hat, labels=classes, average="macro", zero_division=0))

            rows.append({
                "mode": mode,
                "n_components": n_comp,
                "cv_mean_recall_c3_c4": float(np.mean(fold_c34)),
                "cv_std_recall_c3_c4": float(np.std(fold_c34)),
                "cv_mean_macro_recall": float(np.mean(fold_macro)),
                "cv_std_macro_recall": float(np.std(fold_macro)),
            })

    cv_df = pd.DataFrame(rows).sort_values(
        by=["cv_mean_recall_c3_c4", "cv_mean_macro_recall"],
        ascending=False,
    ).reset_index(drop=True)
    cv_df.to_csv(paths["PLS_CV_RESULTS_CSV"], index=False)

    best = cv_df.iloc[0]
    best_use_vip = best["mode"] == "VIP"
    best_n_comp = int(best["n_components"])

    _, final_model = fit_predict_plsda(
        X_train,
        y_train,
        X_test,
        n_components=best_n_comp,
        use_vip=best_use_vip,
        n_classes=n_classes,
        seed=RANDOM_STATE,
    )

    sel = final_model["selected_idx"]
    X_test_use = X_test[:, sel]
    X_test_s = final_model["scaler"].transform(X_test_use)
    scores_test = final_model["pls"].predict(X_test_s)
    y_pred = np.argmax(scores_test, axis=1)

    acc = accuracy_score(y_test, y_pred)
    recalls = recall_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
    report = classification_report(
        y_test,
        y_pred,
        labels=classes,
        target_names=[f"Cluster_{c}" for c in class_names],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    pred_df = merged.loc[idx_test, ["File X", "File Y"]].copy()
    pred_df["True_Label"] = le.inverse_transform(y_test)
    pred_df["Pred_Label"] = le.inverse_transform(y_pred)
    pred_df.to_csv(paths["PLS_TEST_PRED_CSV"], index=False)

    fig, ax = plt.subplots(figsize=(6.8, 5.8), facecolor="#f8f8f8")
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Tuned PLS-DA Confusion Matrix", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
    save_fig(paths["PLS_CM_PNG"])

    if best_use_vip and final_model["vip_scores"] is not None:
        vip_df = pd.DataFrame({"Band": band_cols, "VIP": final_model["vip_scores"]}).sort_values("VIP", ascending=False)
        vip_df.to_csv(paths["PLS_VIP_CSV"], index=False)

    model_bundle = {
        "scaler": final_model["scaler"],
        "pls_model": final_model["pls"],
        "selected_idx": final_model["selected_idx"],
        "selected_bands": [band_cols[i] for i in final_model["selected_idx"]],
        "use_vip": best_use_vip,
        "n_components": best_n_comp,
        "label_encoder": le,
        "class_names": class_names.tolist(),
        "band_cols_all": band_cols,
        "cv_best": best.to_dict(),
    }
    joblib.dump(model_bundle, paths["PLS_MODEL_FILE"])

    with open(paths["PLS_METRICS_TXT"], "w", encoding="utf-8") as f:
        f.write("Improved PLS-DA with tuning + balancing + optional VIP\n")
        f.write(f"Samples: {len(y)}\n")
        f.write(f"Train/Test split: {1 - TEST_SIZE:.2f}/{TEST_SIZE:.2f}\n")
        f.write(f"Best mode: {'VIP' if best_use_vip else 'FULL'}\n")
        f.write(f"Best n_components: {best_n_comp}\n")
        f.write(f"Selected bands: {len(final_model['selected_idx'])}\n")
        f.write(f"Accuracy: {acc:.6f}\n\n")
        f.write(report)

    return model_bundle



def stage_4_apply_model(df_smooth: pd.DataFrame, model: dict, band_cols: list[str], paths: dict[str, str]) -> tuple[pd.DataFrame, int]:
    selected_bands = model["selected_bands"]
    X = df_smooth[selected_bands].to_numpy(float)
    X_scaled = model["scaler"].transform(X)

    score_raw = model["pls_model"].predict(X_scaled)
    score_soft = softmax(score_raw)
    pred_encoded = np.argmax(score_raw, axis=1)
    pred_label = model["label_encoder"].inverse_transform(pred_encoded)
    class_names = model["class_names"]

    pred_df = df_smooth[["File X", "File Y"]].copy()
    pred_df["Pred_Class"] = pred_label
    for j, cname in enumerate(class_names):
        pred_df[f"Score_Class_{cname}"] = score_raw[:, j]
    for j, cname in enumerate(class_names):
        pred_df[f"SoftScore_Class_{cname}"] = score_soft[:, j]
    pred_df.to_csv(paths["MODULE_PRED_CSV"], index=False)

    x = df_smooth["File X"].astype(int).to_numpy()
    y = df_smooth["File Y"].astype(int).to_numpy()
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    h = y_max - y_min + 1
    w = x_max - x_min + 1

    n_classes = len(class_names)
    fig_cols = 3
    fig_rows = int(np.ceil(n_classes / fig_cols))
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(5.4 * fig_cols, 4.8 * fig_rows), facecolor="#f8f8f8")
    axes = np.atleast_1d(axes).ravel()
    for j, cname in enumerate(class_names):
        grid = np.full((h, w), np.nan)
        grid[y - y_min, x - x_min] = score_soft[:, j]
        im = axes[j].imshow(grid, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        axes[j].set_title(f"Class {cname} score")
        axes[j].set_xticks([])
        axes[j].set_yticks([])
        fig.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04).set_label("Soft score")
    for j in range(n_classes, len(axes)):
        axes[j].axis("off")
    fig.suptitle("PLS-DA Class-Score Images", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(paths["CLASS_SCORE_MAPS_PNG"], dpi=180)
    plt.close()

    class_grid = np.full((h, w), np.nan)
    class_grid[y - y_min, x - x_min] = pred_label
    plt.figure(figsize=(8.5, 7), facecolor="#f8f8f8")
    im = plt.imshow(class_grid, cmap="tab10", interpolation="nearest", vmin=min(class_names), vmax=max(class_names))
    plt.title("PLS-DA Final Class Map", fontsize=13, fontweight="bold")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im, fraction=0.046, pad=0.04).set_label("Predicted class")
    save_fig(paths["FINAL_CLASS_MAP_PNG"])

    summary = pd.read_csv(paths["KMEANS_SUMMARY_CSV"])
    crack_class = int(summary.loc[summary["Mean_AllBands"].idxmin(), "Cluster_Label"])
    crack_mask = (pred_label == crack_class).astype(int)
    crack_grid = np.full((h, w), np.nan)
    crack_grid[y - y_min, x - x_min] = crack_mask
    n_total = len(pred_label)
    n_crack = int(crack_mask.sum())
    crack_pct = 100.0 * n_crack / n_total

    plt.figure(figsize=(8.5, 7), facecolor="#f8f8f8")
    plt.imshow(crack_grid, cmap="gray_r", interpolation="nearest", vmin=0, vmax=1)
    plt.title(f"Crack Map (class {crack_class}) | extent = {n_crack}/{n_total} ({crack_pct:.2f}%)", fontsize=12, fontweight="bold")
    plt.xticks([])
    plt.yticks([])
    save_fig(paths["CRACK_MAP_PNG"])

    with open(paths["CRACK_SUMMARY_TXT"], "w", encoding="utf-8") as f:
        f.write("PLS-DA crack extent summary\n")
        f.write(f"Chosen crack class: {crack_class}\n")
        f.write(f"Total pixels: {n_total}\n")
        f.write(f"Crack pixels: {n_crack}\n")
        f.write(f"Crack extent (%): {crack_pct:.4f}\n")

    return pred_df, crack_class



def stage_5_component_maps(df_smooth: pd.DataFrame, model: dict, paths: dict[str, str]) -> pd.DataFrame:
    selected_bands = model["selected_bands"]
    X = df_smooth[selected_bands].to_numpy(float)
    X_scaled = model["scaler"].transform(X)
    scores = model["pls_model"].transform(X_scaled)

    out = df_smooth[["File X", "File Y"]].copy()
    for i in range(scores.shape[1]):
        out[f"PLS_Component_{i + 1}"] = scores[:, i]
    out.to_csv(paths["PLS_COMPONENT_CSV"], index=False)

    x = df_smooth["File X"].astype(int).to_numpy()
    y = df_smooth["File Y"].astype(int).to_numpy()
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    h = y_max - y_min + 1
    w = x_max - x_min + 1

    n_comp = scores.shape[1]
    ncols = 3
    nrows = int(np.ceil(n_comp / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.8 * nrows), facecolor="#f8f8f8")
    axes = np.atleast_1d(axes).ravel()

    for i in range(n_comp):
        grid = np.full((h, w), np.nan)
        grid[y - y_min, x - x_min] = scores[:, i]
        vmax = np.nanpercentile(np.abs(grid), 98)
        if vmax == 0:
            vmax = 1.0
        im = axes[i].imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        axes[i].set_title(f"PLS Component {i + 1}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04).set_label("Score")

    for i in range(n_comp, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Spatial Maps of PLS Components (Per Pixel)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(paths["PLS_COMPONENT_MAPS_PNG"], dpi=180)
    plt.close()
    return out



def stage_6_component1_only(df_comp: pd.DataFrame, paths: dict[str, str]) -> None:
    x = df_comp["File X"].astype(int).to_numpy()
    y = df_comp["File Y"].astype(int).to_numpy()
    values = df_comp["PLS_Component_1"].to_numpy(float)

    grid, _, _, _, _ = build_xy_grid(x, y, values)
    vmax = np.nanpercentile(np.abs(grid), 98)
    if vmax == 0:
        vmax = 1.0

    plt.figure(figsize=(7.2, 6.2), facecolor="#f8f8f8")
    im = plt.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    plt.title("PLS Component 1", fontsize=13, fontweight="bold")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im, fraction=0.046, pad=0.04).set_label("Component 1 score")
    save_fig(paths["COMPONENT1_ONLY_PNG"])



def main() -> None:
    args = parse_args()
    paths = build_output_paths(args.input_csv)

    print("Stage 1/6: Savitzky-Golay smoothing")
    df_smooth, band_cols = stage_1_savgol(paths)

    print("Stage 2/6: K-means pseudo-label generation")
    df_labels = stage_2_kmeans(df_smooth, band_cols, paths)

    print("Stage 3/6: Tuned PLS-DA training")
    model = stage_3_train_plsda(df_smooth, df_labels, band_cols, paths)

    print("Stage 4/6: Apply trained PLS-DA to full module")
    _, crack_class = stage_4_apply_model(df_smooth, model, band_cols, paths)

    print("Stage 5/6: PLS latent component maps")
    df_comp = stage_5_component_maps(df_smooth, model, paths)

    print("Stage 6/6: Final Component 1 image")
    stage_6_component1_only(df_comp, paths)

    print("\nWorkflow complete.")
    print(f"Raw input                 : {paths['RAW_CSV']}")
    print(f"Output folder             : {paths['OUTPUT_DIR']}")
    print(f"Smoothed spectra          : {paths['SMOOTH_CSV']}")
    print(f"K-means labels            : {paths['KMEANS_LABEL_CSV']}")
    print(f"Trained PLS-DA model      : {paths['PLS_MODEL_FILE']}")
    print(f"Module predictions        : {paths['MODULE_PRED_CSV']}")
    print(f"PLS component scores      : {paths['PLS_COMPONENT_CSV']}")
    print(f"Final Component 1 image   : {paths['COMPONENT1_ONLY_PNG']}")
    print(f"Crack class (proxy)       : {crack_class}")


if __name__ == "__main__":
    main()
