import os
import sqlite3
import numpy as np
import pandas as pd
from usearch.index import Index

# =====================================================
# CONFIG
# =====================================================

DB_PATH = r"D:\bird\embedding\hoplite.sqlite"

USEARCH_PATH = r"D:\bird\embedding\usearch.index"

TRAIN_CSV = r"D:\bird\train.csv"

OUTPUT_DIR = r"D:\bird\merged_parquet"

BATCH_SIZE = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)
# =====================================================
# LOAD TRAIN CSV
# =====================================================

train_df = pd.read_csv(TRAIN_CSV)

print(train_df.head())

# =====================================================
# IMPORTANT
# =====================================================
# train.csv must contain:
#
# filename
# scientific_name
#
# For example:
#
# filename                scientific_name
# iNat1114648.ogg         Turdus_merula
#
# =====================================================

train_df["filename"] = train_df["filename"].astype(str)

# =====================================================
# LOAD USEARCH INDEX
# =====================================================

usearch_idx = Index.restore(USEARCH_PATH, view=True)

print("USearch index size:", len(usearch_idx))

# =====================================================
# CONNECT SQLITE
# =====================================================

conn = sqlite3.connect(DB_PATH)

# =====================================================
# TOTAL ROWS
# =====================================================

count_query = """
SELECT COUNT(*)
FROM hoplite_embeddings
"""

total_rows = pd.read_sql_query(
    count_query,
    conn
).iloc[0, 0]

print("Total embeddings:", total_rows)

# =====================================================
# BATCH LOOP
# =====================================================

offset = 0
batch_idx = 0

while offset < total_rows:
    print(f"\nProcessing {offset} / {total_rows}")

    query = f"""
    SELECT
        e.id,
        e.source_idx,
        s.source
    FROM hoplite_embeddings e
    JOIN hoplite_sources s
        ON e.source_idx = s.id
    LIMIT {BATCH_SIZE}
    OFFSET {offset}
    """

    batch_df = pd.read_sql_query(
        query,
        conn
    )

    # =================================================
    # EXTRACT FILENAME
    # =================================================

    batch_df["filename"] = batch_df["source"].apply(
        lambda x: (
        os.path.join(
            os.path.basename(os.path.dirname(x)),
            os.path.basename(x)
        ).replace("\\", "/")
    )
    )
     # =================================================
    # MERGE LABELS
    # =================================================

    merged_df = batch_df.merge(
        train_df[["filename", "scientific_name"]],
        on="filename",
        how="left"
    )
    # =================================================
    # REMOVE UNMATCHED
    # =================================================

    merged_df = merged_df.dropna(
        subset=["scientific_name"]
    )

    # =================================================
    # FETCH EMBEDDINGS FROM USEARCH
    # =================================================

    ids = merged_df["id"].values

    embeddings = np.stack(
        usearch_idx.get(ids)
    ).astype(np.float32)

    print("Embedding shape:", embeddings.shape)
     # =================================================
    # CREATE FEATURE DF
    # =================================================

    feature_df = pd.DataFrame(
        embeddings,
        columns=[
            f"f_{i}"
            for i in range(embeddings.shape[1])
        ]
    )
 # =================================================
    # ADD METADATA
    # =================================================

    feature_df["label"] = merged_df[
        "scientific_name"
    ].values

    feature_df["filename"] = merged_df[
        "filename"
    ].values

    feature_df["source"] = merged_df[
        "source"
    ].values

    feature_df["source_idx"] = merged_df[
        "source_idx"
    ].values
 # =================================================
    # SAVE PARQUET
    # =================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        f"batch_{batch_idx:04d}.parquet"
    )

    feature_df.to_parquet(
        output_path,
        index=False
    )

    print("Saved:", output_path)

    offset += BATCH_SIZE
    batch_idx += 1
print("\nDONE")