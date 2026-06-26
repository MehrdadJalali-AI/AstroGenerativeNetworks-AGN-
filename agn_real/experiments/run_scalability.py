from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from agn_real.data import load_real_dataset
from agn_real.splits import stratified_node_holdout
from agn_real.utils import apply_feature_mode, ensure_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Record scalability metadata for real datasets.")
    parser.add_argument("--datasets", nargs="+", default=["PubMed", "CoauthorCS", "CoauthorPhysics", "AmazonComputers", "AmazonPhoto"])
    parser.add_argument("--data_root", default="data/real")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--holdout_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rows = []
    for dataset in args.datasets:
        row = {"dataset": dataset}
        start = time.perf_counter()
        try:
            data, meta = load_real_dataset(dataset, args.data_root)
            data = apply_feature_mode(data, "raw_plus_structural")
            split_start = time.perf_counter()
            split = stratified_node_holdout(data, args.holdout_ratio, args.seed, "degree_stratified")
            row.update(meta)
            row["status"] = "ok"
            row["load_time_sec"] = split_start - start
            row["split_time_sec"] = time.perf_counter() - split_start
            row["observed_nodes"] = split.observed_data.num_nodes
            row["hidden_nodes"] = len(split.hidden_node_ids)
            row["memory_usage_note"] = "Use system monitor/psutil if installed; not measured in this minimal run."
        except Exception as exc:
            row["status"] = "not_run"
            row["reason"] = str(exc)
        rows.append(row)
    out = ensure_dir(Path(args.output_dir) / "summary")
    pd.DataFrame(rows).to_csv(out / "scalability.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

