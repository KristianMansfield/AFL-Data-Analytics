#!/usr/bin/env python3

# Positional roles with HDBSCAN, a density-based method that decides the number of roles itself.
#
# What it does: looks for regions where player-seasons are packed densely together, separated by
# sparser gaps, and calls each dense region a role. It never takes a number of roles: that falls
# out of the data, controlled only by the smallest group size worth calling a role
# (--min-cluster-size). Players who don't sit in any dense region are labelled noise rather than
# being forced into the nearest role. The script first shows how the number of roles and the share
# of noise change across a range of minimum group sizes, then fits the chosen one.
#
# Why it matters for the analysis: k-means and the Gaussian mixture put every player in a role,
# even players who don't really fit one. HDBSCAN is an independent opinion on how many roles exist
# that doesn't depend on a scoring rule, and its noise points are the genuinely unusual players:
# one-off hybrids and specialists who might deserve their own treatment when predicting awards.
# Density methods struggle in many dimensions, so it uses fewer principal components by default
# (--pca-variance 0.5). Apart from rucks, AFL roles blend into each other rather than being
# separated by empty space, so expect a large share of noise: that in itself is evidence that
# soft (probabilistic) role assignments suit this data better than hard ones.
#
# Usage: ../venv/bin/python hdbscanClustering.py --years 2021-2025 [--min-cluster-size 40]

import os

import pandas as pd
from sklearn.cluster import HDBSCAN

import seasonFeatures


def main():
    parser = seasonFeatures.newParser("Positional roles with HDBSCAN")
    parser.add_argument(
        "--min-cluster-size", type=int, default=15,
        help="Smallest group of player-seasons that counts as a role (default 15)",
    )
    parser.add_argument(
        "--min-samples", type=int, default=5,
        help="Neighbours needed for a player to count as being in a dense region; higher "
        "marks more players as noise (default 5)",
    )
    parser.add_argument(
        "--sweep", nargs="*", type=int, default=[10, 20, 30, 40, 60, 80, 100, 150],
        help="Minimum group sizes to compare before fitting",
    )
    parser.add_argument(
        "--selection", choices=["eom", "leaf"], default="leaf",
        help="'leaf' gives more, finer roles; 'eom' fewer, broader ones (default leaf)",
    )
    parser.set_defaults(pca_variance=0.5)
    args = parser.parse_args()

    info, z = seasonFeatures.prepare(args)
    space = seasonFeatures.reduce(z, args.pca_variance, args.seed)
    out_dir = seasonFeatures.outputDirectory("hdbscan", args)

    def fit(size):
        return HDBSCAN(min_cluster_size=size, min_samples=args.min_samples,
                       cluster_selection_method=args.selection, copy=True).fit(space)

    if args.sweep:
        rows = []
        for size in args.sweep:
            labels = fit(size).labels_
            rows.append({"min_cluster_size": size,
                         "roles": len(set(labels) - {-1}),
                         "noise": (labels == -1).mean()})
        sweep = pd.DataFrame(rows).set_index("min_cluster_size")
        sweep.to_csv(os.path.join(out_dir, "sweep.csv"))
        print("\nRoles found by minimum group size:")
        print(sweep.to_string(formatters={"noise": "{:.0%}".format}))

    model = fit(args.min_cluster_size)
    labels = model.labels_
    print(f"\nHDBSCAN with min cluster size {args.min_cluster_size}: "
          f"{len(set(labels) - {-1})} roles, {(labels == -1).mean():.0%} of players unclustered")

    seasonFeatures.describeClusters(info, z, labels, space, out_dir)
    seasonFeatures.compareToListedPositions(info, labels, out_dir)

    # Players in a role but only weakly (membership strength 0 to 1)
    membership = pd.read_csv(os.path.join(out_dir, "assignments.csv"))
    membership["strength"] = model.probabilities_
    membership.to_csv(os.path.join(out_dir, "assignments.csv"), index=False)
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
