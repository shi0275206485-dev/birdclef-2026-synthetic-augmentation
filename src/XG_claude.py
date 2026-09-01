import pyarrow.dataset as ds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score,
    f1_score,
)
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier

# =====================================================
# LOAD PARQUET FILES
# =====================================================

PARQUET_DIR = r"D:\bird\merged_parquet"

dataset = ds.dataset(PARQUET_DIR, format="parquet")

# Only read the needed columns, skip source / source_idx
schema_names = dataset.schema.names
feature_cols = [c for c in schema_names if c.startswith("f_")]
needed_cols = feature_cols + ["label", "filename"]

print("Num parquet:", len(dataset.files))
print("Loading", len(needed_cols), "columns ...")

table = dataset.to_table(columns=needed_cols)
df = table.to_pandas()
del table  # free PyArrow memory before building numpy array

print("Data shape:", df.shape)

# =====================================================
# FEATURES
# =====================================================

X = df[feature_cols].to_numpy(dtype=np.float32)
print("Number of features (dimensions):", len(feature_cols))
print("First few column names:", feature_cols[:5])
print("Value range: min=%.3f, max=%.3f, mean=%.3f, std=%.3f" % (
    X.min(), X.max(), X.mean(), X.std()))
print("First 10 values of a single sample:", X[0, :10])
# =====================================================
# LABEL
# =====================================================

le = LabelEncoder()
y = le.fit_transform(df["label"])
num_class = len(le.classes_)
print("Num classes:", num_class)

# =====================================================
# CLASS DISTRIBUTION  (get a clear picture of the imbalance first)
# =====================================================

print("\n========== CLASS DISTRIBUTION ==========")
counts = pd.Series(y).value_counts().sort_values()
print("Total samples:", len(y))
print("Smallest 10 classes (encoded_id: count):")
for cid, cnt in counts.head(10).items():
    print(f"  {le.classes_[cid]:<30} {cnt}")
print("Largest 5 classes:")
for cid, cnt in counts.tail(5).items():
    print(f"  {le.classes_[cid]:<30} {cnt}")
print("Imbalance ratio (max/min):", counts.max() / counts.min())

# =====================================================
# GROUP SPLIT  (grouped by filename, to avoid leakage)
# =====================================================

groups = df["filename"]

gss = GroupShuffleSplit(
    test_size=0.2,
    n_splits=1,
    random_state=42,
)

train_idx, val_idx = next(gss.split(X, y, groups))

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

print("\nX_train:", X_train.shape)
print("X_val:  ", X_val.shape)

# =====================================================
# SAMPLE WEIGHT  (handle class imbalance, boost macro F1)
# =====================================================
# class_weight="balanced": each class's weight = n_samples / (n_classes * count)
# Minority classes get higher sample weight, so they carry more weight in the loss.
# Note: weights are computed only on the training set, never on the validation set.

sample_weight = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)
# =====================================================
# Logitc Regression
# =====================================================

print("\n========== LogisticRegression (linear probe) ==========")
clf = SGDClassifier(
    loss="log_loss",         # = logistic regression
    class_weight="balanced",
    max_iter=50,
    n_jobs=4,
    random_state=42,
)
clf.fit(X_train, y_train)

lr_pred = clf.predict(X_val)
print("LogReg Accuracy:   ", round(accuracy_score(y_val, lr_pred), 4))
print("LogReg Macro F1:   ", round(f1_score(y_val, lr_pred, average="macro"), 4))
print("LogReg Weighted F1:", round(f1_score(y_val, lr_pred, average="weighted"), 4))
# =====================================================
# XGBOOST
# =====================================================

model = XGBClassifier(
    objective="multi:softprob",
    num_class=num_class,

    n_estimators=500,          # high ceiling, relying on early stopping to cut it off automatically
    max_depth=6,                # increase model capacity (was 6)
    learning_rate=0.05,         # smaller step size with more trees, generalizes better (was 0.1)

    subsample=0.8,
    colsample_bytree=0.8,

    min_child_weight=3,         # discourage splitting leaves for noisy samples
    reg_lambda=5.0,             # L2 regularization
    reg_alpha=1.0,              # L1 regularization
    gamma=0.2,                  # minimum gain threshold for a split

    max_bin=256,                # lower to 128 if GPU memory is tight

    tree_method="hist",
    device="cuda",              # change to "cpu" if you don't have a GPU
    n_jobs=-1,

    eval_metric="mlogloss",
    early_stopping_rounds=30,   # paired with a small lr, allow more patience
    random_state=42,
)

# =====================================================
# TRAIN
# =====================================================

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weight,     # key: weighted training
    eval_set=[
        (X_train, y_train),
        (X_val, y_val),
    ],
    verbose=True,
)

print("\nBest iteration:", model.best_iteration)
print("Best val mlogloss:", model.best_score)

# =====================================================
# PREDICT & EVAL
# =====================================================

print("\nPredicting...")
pred_prob = model.predict_proba(X_val)
pred = model.predict(X_val)

acc = accuracy_score(y_val, pred)
macro_f1 = f1_score(y_val, pred, average="macro")
weighted_f1 = f1_score(y_val, pred, average="weighted")

print("\n========== METRICS ==========")
print("Accuracy:    ", round(acc, 4))
print("Macro F1:    ", round(macro_f1, 4))
print("Weighted F1: ", round(weighted_f1, 4))

# TOP-5
k = min(5, num_class)
top5 = top_k_accuracy_score(
    y_val,
    pred_prob,
    k=k,
    labels=np.arange(num_class),
)
print(f"Top-{k} Accuracy:", round(top5, 4))

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\n========== CLASSIFICATION REPORT ==========\n")
print(
    classification_report(
        y_val,
        pred,
        labels=np.arange(num_class),
        target_names=le.classes_,
        zero_division=0,
    )
)

# =====================================================
# LOSS CURVE  (key diagnostic for underfitting / overfitting)
# =====================================================

results = model.evals_result()

plt.figure(figsize=(10, 5))
plt.plot(results["validation_0"]["mlogloss"], label="train")
plt.plot(results["validation_1"]["mlogloss"], label="val")
plt.axvline(model.best_iteration, color="gray", linestyle="--",
            label=f"best iter = {model.best_iteration}")
plt.xlabel("Iteration")
plt.ylabel("Log Loss")
plt.title("XGBoost Training Curve")
plt.legend()
plt.tight_layout()
plt.savefig("training_curve.png", dpi=120)
plt.show()

# =====================================================
# CONFUSION MATRIX  (top 20 classes)
# =====================================================

cm = confusion_matrix(y_val, pred)
num_show = min(20, num_class)

plt.figure(figsize=(14, 12))
sns.heatmap(cm[:num_show, :num_show], cmap="Blues")
plt.title("Confusion Matrix (Top 20 Classes)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.show()

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = model.feature_importances_
idx = np.argsort(importance)[::-1][:30]

plt.figure(figsize=(12, 6))
plt.bar(range(len(idx)), importance[idx])
plt.xticks(range(len(idx)), [feature_cols[i] for i in idx],
           rotation=90, fontsize=7)
plt.title("Top 30 Feature Importance")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
plt.show()

# =====================================================
# SAVE
# =====================================================

model.save_model("birdnet_xgb_2.json")
print("\nDONE")