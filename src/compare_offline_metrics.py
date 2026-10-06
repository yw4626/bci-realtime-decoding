"""
Compare mean metrics from two offline_metrics.csv files (e.g. single-subject vs multi-subject).
"""

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_csv", required=True, help="e.g. artifacts/offline_metrics.csv")
    parser.add_argument("--other_csv", required=True, help="e.g. artifacts_multi/offline_metrics.csv")
    args = parser.parse_args()

    b = pd.read_csv(args.baseline_csv)
    o = pd.read_csv(args.other_csv)
    bs = b.groupby("family")[["rmse", "mae", "cosine_distance", "pearson_r"]].mean()
    osm = o.groupby("family")[["rmse", "mae", "cosine_distance", "pearson_r"]].mean()
    print("=== baseline (self-train / self-val) ===")
    print(bs)
    print("\n=== other (e.g. multi-train / sub1-val) ===")
    print(osm)
    print("\n=== difference (other - baseline) ===")
    print(osm - bs)


if __name__ == "__main__":
    main()
