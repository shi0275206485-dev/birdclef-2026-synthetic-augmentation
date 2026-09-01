"""
Recording-level aggregated evaluation.

Background: BirdNET embeddings are segment-level (each recording is cut into multiple
3-second segments), but labels are weak recording-level labels (one label per recording,
copied to all its segments). Many segments actually contain no target bird call
(wind noise / silence), so segment-level evaluation is affected by label noise.

Approach: aggregate the predicted probabilities of all segments belonging to the same
recording (filename) into a single recording-level prediction, then compare against the
recording-level ground-truth label. Both mean and max aggregation are provided for comparison.

Precondition: the validation set must be split entirely by filename (segments from the
same recording never cross train/val), which compliant_split / GroupShuffleSplit
(grouped by filename) already guarantees.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score


def aggregate_to_recording(pred_prob, filenames, y_segment, num_class):
    """
    Aggregate segment-level probabilities to recording-level.

    pred_prob  : (n_segments, num_class) segment-level predicted probabilities
    filenames  : (n_segments,) recording each segment belongs to
    y_segment  : (n_segments,) segment-level ground-truth labels (identical within a recording)
    Returns: rec_ids, y_rec (recording ground-truth labels), prob_mean, prob_max
    """
    filenames = np.asarray(filenames)
    y_segment = np.asarray(y_segment)

    rec_ids = np.unique(filenames)
    prob_mean = np.zeros((len(rec_ids), num_class), dtype=np.float64)
    prob_max = np.zeros((len(rec_ids), num_class), dtype=np.float64)
    y_rec = np.zeros(len(rec_ids), dtype=np.int64)

    for i, rec in enumerate(rec_ids):
        m = filenames == rec
        p = pred_prob[m]
        prob_mean[i] = p.mean(axis=0)        # mean-probability aggregation
        prob_max[i] = p.max(axis=0)          # max-probability aggregation
        # recording ground-truth label = the label of any segment from that recording
        # (identical across segments under weak labeling)
        y_rec[i] = y_segment[m][0]

    return rec_ids, y_rec, prob_mean, prob_max


def evaluate_recording_level(pred_prob, filenames, y_segment, num_class,
                             class_names=None, verbose=True):
    """
    Full recording-level evaluation, printing metrics for both mean and max aggregation.
    Returns a dict for easier downstream comparison.
    """
    rec_ids, y_rec, prob_mean, prob_max = aggregate_to_recording(
        pred_prob, filenames, y_segment, num_class
    )

    results = {}
    labels_all = np.arange(num_class)

    for name, prob in [("mean", prob_mean), ("max", prob_max)]:
        pred = prob.argmax(axis=1)
        acc = accuracy_score(y_rec, pred)
        macro = f1_score(y_rec, pred, average="macro", zero_division=0)
        weighted = f1_score(y_rec, pred, average="weighted", zero_division=0)
        k = min(5, num_class)
        try:
            top5 = top_k_accuracy_score(y_rec, prob, k=k, labels=labels_all)
        except Exception:
            top5 = float("nan")
        results[name] = dict(acc=acc, macro_f1=macro,
                             weighted_f1=weighted, top5=top5)

    if verbose:
        print("\n" + "=" * 55)
        print(f"Recording-level evaluation  (recordings: {len(rec_ids)}, segments: {len(pred_prob)})")
        print("=" * 55)
        for name in ("mean", "max"):
            r = results[name]
            print(f"\n[{name} aggregation]")
            print(f"  Accuracy:    {r['acc']:.4f}")
            print(f"  Macro F1:    {r['macro_f1']:.4f}")
            print(f"  Weighted F1: {r['weighted_f1']:.4f}")
            print(f"  Top-5 Acc:   {r['top5']:.4f}")
        print("\nNote: mean aggregation is usually more robust; max is more sensitive to "
              "recordings where only a few segments contain bird calls.")

    return results, (rec_ids, y_rec, prob_mean, prob_max)