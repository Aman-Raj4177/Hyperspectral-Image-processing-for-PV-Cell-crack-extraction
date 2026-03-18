"""
Improved PLS-DA for k-means labels with:
1) Component tuning (stratified CV)
2) Class balancing (random oversampling on train folds)
3) Optional VIP-based band selection

Primary optimization target: maximize mean recall of clusters 3 and 4.
Secondary target: macro recall.

Outputs:
- plsda_tuned_model.joblib
- plsda_tuned_test_predictions.csv
- plsda_tuned_confusion_matrix.png
- plsda_tuned_metrics.txt
- plsda_cv_results.csv
- plsda_vip_scores.csv (if VIP branch wins)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import recall_score, classification_report, confusion_matrix, accuracy_score
import joblib

BASE = (r"c:\Users\amanraj.ASURITE\ASU Dropbox\Aman Raj\Aman Raj"
        r"\PV Reliability (HSI)\Analysis_02022026_Tripod_PV_field"
        r"\Crack Detection_0317")

SMOOTH_CSV = os.path.join(BASE, "VNIR_1x1_SavGol_smoothed.csv")
LABEL_CSV = os.path.join(BASE, "VNIR_1x1_SavGol_kmeans_k5_labels.csv")

MODEL_FILE = os.path.join(BASE, "plsda_tuned_model.joblib")
PRED_CSV = os.path.join(BASE, "plsda_tuned_test_predictions.csv")
CM_PNG = os.path.join(BASE, "plsda_tuned_confusion_matrix.png")
METRICS_TXT = os.path.join(BASE, "plsda_tuned_metrics.txt")
CV_RESULTS_CSV = os.path.join(BASE, "plsda_cv_results.csv")
VIP_CSV = os.path.join(BASE, "plsda_vip_scores.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.25
N_SPLITS = 5
COMPONENT_GRID = list(range(2, 11))
VIP_THRESHOLD = 1.0


def oversample_to_max_class(X, y, seed=42):
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    idx_all = []
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) < max_count:
            extra = rng.choice(idx, size=max_count - len(idx), replace=True)
            idx = np.concatenate([idx, extra])
        idx_all.append(idx)

    idx_bal = np.concatenate(idx_all)
    rng.shuffle(idx_bal)
    return X[idx_bal], y[idx_bal]


def one_hot(y, n_classes):
    return np.eye(n_classes)[y]


def compute_vip(pls, X_scaled, Y_onehot):
    # VIP scores for PLS: one score per predictor band.
    T = pls.x_scores_            # (n, h)
    W = pls.x_weights_           # (p, h)
    Q = pls.y_loadings_          # (m, h)

    p = W.shape[0]
    h = W.shape[1]

    # Explained contribution per component for Y-space
    s = np.zeros(h)
    for a in range(h):
        t_a = T[:, a]
        q_a = Q[:, a]
        s[a] = np.sum(t_a ** 2) * np.sum(q_a ** 2)

    total_s = np.sum(s)
    if total_s <= 0:
        return np.ones(p)

    vip = np.zeros(p)
    for j in range(p):
        num = 0.0
        for a in range(h):
            w_a = W[:, a]
            denom = np.sum(w_a ** 2)
            if denom > 0:
                num += s[a] * ((W[j, a] ** 2) / denom)
        vip[j] = np.sqrt(p * num / total_s)

    return vip


def fit_predict_plsda(X_train, y_train, X_val, n_components, use_vip, n_classes, seed=42):
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
        vip_scores = compute_vip(pls, X_bal, Y_bal)
        selected_idx = np.where(vip_scores >= VIP_THRESHOLD)[0]

        # Keep a safe minimum if threshold is too strict
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

    model_bundle = {
        "scaler": scaler,
        "pls": pls,
        "selected_idx": selected_idx,
        "use_vip": use_vip,
        "vip_scores": vip_scores,
        "n_components": n_components,
    }
    return y_pred, model_bundle


# Load and prepare dataset
sm = pd.read_csv(SMOOTH_CSV)
lb = pd.read_csv(LABEL_CSV)
sm.columns = sm.columns.str.strip()
lb.columns = lb.columns.str.strip()

merged = sm.merge(lb, on=["File X", "File Y"], how="inner")
band_cols = [c for c in merged.columns if c.startswith("B")]

X = merged[band_cols].to_numpy(float)
y_raw = merged["Cluster_Label"].to_numpy()

le = LabelEncoder()
y = le.fit_transform(y_raw)
classes = np.unique(y)
class_names = le.inverse_transform(classes)
n_classes = len(classes)

# Explicit indices for cluster 3 and 4 in encoded label space
target_clusters = [3, 4]
if not set(target_clusters).issubset(set(class_names.tolist())):
    raise ValueError("Expected cluster labels 3 and 4 to exist in the data.")
idx_c3 = int(np.where(class_names == 3)[0][0])
idx_c4 = int(np.where(class_names == 4)[0][0])

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(y)),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# CV tuning
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
rows = []

for use_vip in [False, True]:
    mode = "VIP" if use_vip else "FULL"
    for n_comp in COMPONENT_GRID:
        fold_c34 = []
        fold_macro = []

        for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_va, y_va = X_train[va_idx], y_train[va_idx]

            y_hat, _ = fit_predict_plsda(
                X_tr, y_tr, X_va,
                n_components=n_comp,
                use_vip=use_vip,
                n_classes=n_classes,
                seed=RANDOM_STATE + fold_id
            )

            recalls = recall_score(y_va, y_hat, labels=classes, average=None, zero_division=0)
            r3 = recalls[idx_c3]
            r4 = recalls[idx_c4]
            macro = recall_score(y_va, y_hat, labels=classes, average="macro", zero_division=0)

            fold_c34.append((r3 + r4) / 2.0)
            fold_macro.append(macro)

        rows.append({
            "mode": mode,
            "n_components": n_comp,
            "cv_mean_recall_c3_c4": float(np.mean(fold_c34)),
            "cv_std_recall_c3_c4": float(np.std(fold_c34)),
            "cv_mean_macro_recall": float(np.mean(fold_macro)),
            "cv_std_macro_recall": float(np.std(fold_macro)),
        })

cv_df = pd.DataFrame(rows)
cv_df = cv_df.sort_values(
    by=["cv_mean_recall_c3_c4", "cv_mean_macro_recall"],
    ascending=False
).reset_index(drop=True)
cv_df.to_csv(CV_RESULTS_CSV, index=False)

best = cv_df.iloc[0]
best_use_vip = (best["mode"] == "VIP")
best_n_comp = int(best["n_components"])

# Final fit on full training split with best config
_, final_model = fit_predict_plsda(
    X_train, y_train, X_test,
    n_components=best_n_comp,
    use_vip=best_use_vip,
    n_classes=n_classes,
    seed=RANDOM_STATE
)

# Predict test with final model
sel = final_model["selected_idx"]
X_test_use = X_test[:, sel]
X_test_s = final_model["scaler"].transform(X_test_use)
scores_test = final_model["pls"].predict(X_test_s)
y_pred = np.argmax(scores_test, axis=1)

acc = accuracy_score(y_test, y_pred)
recalls = recall_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
recall_c3 = recalls[idx_c3]
recall_c4 = recalls[idx_c4]
macro_recall = recall_score(y_test, y_pred, labels=classes, average="macro", zero_division=0)

report = classification_report(
    y_test,
    y_pred,
    labels=classes,
    target_names=[f"Cluster_{c}" for c in class_names],
    digits=4,
    zero_division=0
)
cm = confusion_matrix(y_test, y_pred, labels=classes)

# Save predictions with coordinates
pred_df = merged.loc[idx_test, ["File X", "File Y"]].copy()
pred_df["True_Label"] = le.inverse_transform(y_test)
pred_df["Pred_Label"] = le.inverse_transform(y_pred)
pred_df.to_csv(PRED_CSV, index=False)

# Save confusion matrix
fig, ax = plt.subplots(figsize=(6.8, 5.8), facecolor="#f8f8f8")
im = ax.imshow(cm, cmap="Blues")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Tuned PLS-DA Confusion Matrix", fontsize=11, fontweight="bold")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")

ticks = np.arange(n_classes)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

for i in range(n_classes):
    for j in range(n_classes):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(CM_PNG, dpi=180)
plt.close()

# Save VIP scores if used
if best_use_vip and final_model["vip_scores"] is not None:
    vip_scores = final_model["vip_scores"]
    vip_df = pd.DataFrame({
        "Band": band_cols,
        "VIP": vip_scores
    }).sort_values("VIP", ascending=False)
    vip_df.to_csv(VIP_CSV, index=False)

# Save model bundle
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
joblib.dump(model_bundle, MODEL_FILE)

# Metrics text
with open(METRICS_TXT, "w", encoding="utf-8") as f:
    f.write("Improved PLS-DA with tuning + balancing + optional VIP\n")
    f.write(f"Samples: {len(y)}\n")
    f.write(f"Train/Test split: {1 - TEST_SIZE:.2f}/{TEST_SIZE:.2f}\n")
    f.write(f"Best mode: {'VIP' if best_use_vip else 'FULL'}\n")
    f.write(f"Best n_components: {best_n_comp}\n")
    f.write(f"Selected bands: {len(final_model['selected_idx'])}\n")
    f.write("\nCV best row:\n")
    f.write(best.to_string())
    f.write("\n\nTest metrics:\n")
    f.write(f"Accuracy: {acc:.6f}\n")
    f.write(f"Recall Cluster 3: {recall_c3:.6f}\n")
    f.write(f"Recall Cluster 4: {recall_c4:.6f}\n")
    f.write(f"Macro Recall: {macro_recall:.6f}\n\n")
    f.write("Classification report:\n")
    f.write(report)

print("Improved PLS-DA training complete")
print(f"Best mode         : {'VIP' if best_use_vip else 'FULL'}")
print(f"Best n_components : {best_n_comp}")
print(f"Selected bands    : {len(final_model['selected_idx'])}")
print(f"Test accuracy     : {acc:.4f}")
print(f"Recall Cluster 3  : {recall_c3:.4f}")
print(f"Recall Cluster 4  : {recall_c4:.4f}")
print(f"Macro recall      : {macro_recall:.4f}")
print("\nClassification report:")
print(report)
print("Saved:")
print(f"  {MODEL_FILE}")
print(f"  {PRED_CSV}")
print(f"  {CM_PNG}")
print(f"  {METRICS_TXT}")
print(f"  {CV_RESULTS_CSV}")
if best_use_vip:
    print(f"  {VIP_CSV}")
