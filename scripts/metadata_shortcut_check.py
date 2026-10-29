"""How well can the class be predicted from recording metadata alone (no audio content)?

Features: native sample-rate group, channel count, sample format, clip duration (all from results/eda/audio_info.csv).
Model: random forest, trained and tested with the dataset's own 10-fold split. This measures how much a model could
gain from non-acoustic shortcuts such as clip length (and hence zero padding). Descriptive data analysis only: no
model results are involved. Output: results/eda/metadata_shortcut.json

    python scripts/metadata_shortcut_check.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def main():
    a = pd.read_csv("results/eda/audio_info.csv")
    a["sr_group"] = pd.cut(a.sr, [0, 16000, 32000, 44100, 48000, 96000, 200000],
                           labels=["<=16k", "22-32k", "44.1k", "48k", "96k", "192k"]).astype(str)
    a["short"] = a.duration_s < 3.99
    X = pd.get_dummies(a[["sr_group", "channels", "subtype"]].astype(str))
    X["duration_s"] = a.duration_s
    accs = []
    for k in range(1, 11):
        tr, te = a.fold != k, a.fold == k
        m = RandomForestClassifier(200, random_state=0, n_jobs=1).fit(X[tr], a["class"][tr])
        accs.append(float((m.predict(X[te]) == a["class"][te]).mean()))
    out = {
        "metadata_only_accuracy_mean": float(np.mean(accs)), "per_fold": accs, "chance": 0.1,
        "majority_class_rate": float(a["class"].value_counts(normalize=True).max()),
        "short_clip_share_by_class": (a.groupby("class").short.mean()).round(3).to_dict(),
        "sample_rate_group_share_by_class": pd.crosstab(a["class"], a.sr_group, normalize="index").round(3).to_dict("index"),
        "features": ["native sample-rate group", "channels", "sample format", "duration_s"],
    }
    Path("results/eda/metadata_shortcut.json").write_text(json.dumps(out, indent=2))
    print(f"metadata-only accuracy {out['metadata_only_accuracy_mean']:.3f} (chance 0.10, majority {out['majority_class_rate']:.3f})")


if __name__ == "__main__":
    main()
