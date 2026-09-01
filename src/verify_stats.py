"""
Verify the segment-level numbers in the Data statistics table.
After running, paste the output to the assistant and correct the table in the report accordingly.
"""
import pyarrow.dataset as ds
import numpy as np
import pandas as pd

PARQUET_DIR = r"D:\bird\merged_parquet"   # change to your embedding parquet directory

dataset = ds.dataset(PARQUET_DIR, format="parquet")
# only read the label and filename columns to save memory
table = dataset.to_table(columns=["label", "filename"])
labels = np.asarray(table.column("label").to_numpy(zero_copy_only=False))
filenames = np.asarray(table.column("filename").to_numpy(zero_copy_only=False))

n_seg = len(labels)
n_rec = len(set(filenames))
n_cls = len(set(labels))

# segment-level class distribution
vc = pd.Series(labels).value_counts()
max_cls, max_n = vc.index[0], vc.iloc[0]
min_cls, min_n = vc.index[-1], vc.iloc[-1]

# embedding dimension
fcols = [c for c in dataset.schema.names if c.startswith("f_")]
emb_dim = len(fcols)

print("="*55)
print("Segment-level statistics (for the report's Data statistics table)")
print("="*55)
print(f"Total segments:              {n_seg}")
print(f"Total recordings (unique filename): {n_rec}")
print(f"Number of classes:           {n_cls}")
print(f"Embedding dimension (f_ columns): {emb_dim}")
print(f"Average segments per recording:   {n_seg/n_rec:.1f}")
print()
print(f"Largest class (segments): {max_cls}  ({max_n} segments)")
print(f"Smallest class (segments): {min_cls}  ({min_n} segments)")
print(f"Imbalance ratio (segment-level): {max_n/min_n:.0f}x")
print()
print("The 6 classes with the fewest segments:")
print(vc.tail(6).to_string())
