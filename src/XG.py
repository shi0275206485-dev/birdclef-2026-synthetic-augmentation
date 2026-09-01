import pyarrow.dataset as ds
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score
)
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

print(df.shape)

# =====================================================
# FEATURES
# =====================================================

X = table.select(feature_cols).to_pydict()
X = np.array([X[c] for c in feature_cols], dtype=np.float32).T

# =====================================================
# LABEL
# =====================================================

le = LabelEncoder()

y = le.fit_transform(
    df["label"]
)

# =====================================================
# GROUP SPLIT
# VERY IMPORTANT
# =====================================================

groups = df["filename"]

gss = GroupShuffleSplit(
    test_size=0.2,
    n_splits=1,
    random_state=42
)

train_idx, val_idx = next(
    gss.split(X, y, groups)
)

X_train = X[train_idx]
X_val = X[val_idx]

y_train = y[train_idx]
y_val = y[val_idx]

print(X_train.shape)
print(X_val.shape)

# =====================================================
# XGBOOST
# =====================================================

model = XGBClassifier(

    objective="multi:softprob",

    num_class=len(le.classes_),

    n_estimators=150,

    max_depth=6,

    learning_rate=0.1,

    subsample=0.8,

    colsample_bytree=0.8,

    tree_method="hist",

    device="cuda",  # if you don't have a GPU, change this to "cpu"

    n_jobs=-1,

    eval_metric="mlogloss",

    early_stopping_rounds=20,

    random_state=42
)

# =====================================================
# TRAIN
# =====================================================

model.fit(

    X_train,
    y_train,

    eval_set=[
        (X_train, y_train),
        (X_val, y_val)
    ],

    verbose=True
)
print("\nPredicting...")
# =====================================================
# EVAL
# =====================================================
pred_prob = model.predict_proba(X_val)
pred = model.predict(X_val)

print(
    classification_report(
        y_val,
        pred
    )
)
# =====================================================
# ACCURACY
# =====================================================

acc = accuracy_score(y_val, pred)

print("\nAccuracy:", acc)

# =====================================================
# TOP-5 ACCURACY
# =====================================================

k = min(5, len(le.classes_))

top5 = top_k_accuracy_score(
    y_val,
    pred_prob,
    k=k,
    labels=np.arange(len(le.classes_))
)

print("Top-5 Accuracy:", top5)
# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_val,
        pred,
        labels=np.arange(len(le.classes_)),
        target_names=le.classes_,
        zero_division=0
    )
)
# =====================================================
# LOSS CURVE
# =====================================================

results = model.evals_result()

plt.figure(figsize=(10, 5))

plt.plot(
    results['validation_0']['mlogloss'],
    label='train'
)

plt.plot(
    results['validation_1']['mlogloss'],
    label='val'
)

plt.xlabel('Iteration')
plt.ylabel('Log Loss')
plt.title('XGBoost Training Curve')
plt.legend()

plt.show()
# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_val,
    pred
)

# Only show the top 20 classes
num_show = min(20, len(le.classes_))

plt.figure(figsize=(14, 12))

sns.heatmap(
    cm[:num_show, :num_show],
    cmap='Blues'
)

plt.title('Confusion Matrix (Top 20 Classes)')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.show()
# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = model.feature_importances_

idx = np.argsort(importance)[::-1][:30]

plt.figure(figsize=(12, 6))

plt.bar(
    range(len(idx)),
    importance[idx]
)

plt.title('Top Feature Importance')

plt.xlabel('Feature Index')
plt.ylabel('Importance')

plt.show()
# =====================================================
# SAVE
# =====================================================

model.save_model(
    "birdnet_xgb_" + str(1)
      + ".json"
)

print("DONE")