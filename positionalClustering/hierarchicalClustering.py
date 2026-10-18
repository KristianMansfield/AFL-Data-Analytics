#!/usr/bin/env python3

# Positional roles with hierarchical clustering (Ward linkage), drawn as a dendrogram.
#
# What it does: starts with every player-season as its own group and repeatedly merges the two
# groups whose merger adds the least within-group spread (Ward's method), until everyone is in one
# group. The full history of merges is a tree (the dendrogram); cutting it at different heights
# gives any number of roles. The number of roles is chosen where the merge distance jumps the most
# (the longest vertical gap in the tree) or by the best silhouette score in a range, and the script prints how the roles split from the top of the tree down (e.g. defenders vs
# everyone else, then key vs general defenders, ...).
#
# Why it matters for the analysis: the other methods give a flat list of roles; this one shows
# how the roles relate to each other, which mirrors how positions are actually described
# (forward -> key forward / small forward). Long vertical gaps in the dendrogram show cut points
# the data strongly supports, which is an independent check on the number of roles BIC or the gap
# statistic chose. It is also deterministic: the same data always gives the same tree.
#
# Usage: ../venv/bin/python hierarchicalClustering.py --years 2021-2025 [--k 10]

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import silhouette_score

import seasonFeatures


def topFeatures(z, members, count=3):
    profile = z.iloc[members].mean().sort_values(ascending=False)
    return ", ".join(profile.index[:count])


def printSplits(tree, z, k):
    """How the tree splits from 1 role down to k: which group divides at each step, and what
    the two halves are highest in."""
    print("\nHow the roles split, from the top of the tree:")
    previous = np.ones(len(z), dtype=int)
    for n in range(2, k + 1):
        current = fcluster(tree, n, criterion="maxclust")
        # The parent group is the one whose members now carry more than one label
        for parent in np.unique(previous):
            members = np.flatnonzero(previous == parent)
            children = np.unique(current[members])
            if len(children) > 1:
                halves = [np.flatnonzero(current == child) for child in children]
                described = " | ".join(f"{len(h)}: {topFeatures(z, h)}" for h in halves)
                print(f"  {n} roles: {len(members)} split into  {described}")
        previous = current


def main():
    parser = seasonFeatures.newParser("Positional roles with Ward hierarchical clustering")
    parser.add_argument("--k-min", type=int, default=4, help="Fewest roles to try (default 4)")
    parser.add_argument("--k-max", type=int, default=16, help="Most roles to try (default 16)")
    parser.add_argument("--k", type=int, help="Cut the tree into this many roles")
    parser.add_argument(
        "--select", choices=["jump", "silhouette"], default="jump",
        help="Cut where the merge distance jumps most, or at the best silhouette (default jump)",
    )
    args = parser.parse_args()

    info, z = seasonFeatures.prepare(args)
    space = seasonFeatures.reduce(z, args.pca_variance, args.seed)
    out_dir = seasonFeatures.outputDirectory("hierarchical", args)

    tree = linkage(space, method="ward")

    # Merge heights: a big jump just above a cut means the groups below it are well separated
    heights = tree[::-1, 2]
    jumps = pd.Series(heights[:-1] - heights[1:], index=range(2, len(heights) + 1))
    jump_k = int(jumps.loc[args.k_min : args.k_max].idxmax())
    print(f"\nLargest merge-distance jump in range: cutting into {jump_k} roles")

    if args.k:
        best_k = args.k
    else:
        scores = pd.Series(
            {k: silhouette_score(space, fcluster(tree, k, criterion="maxclust"))
             for k in range(args.k_min, args.k_max + 1)},
            name="silhouette",
        )
        scores.index.name = "k"
        scores.to_csv(os.path.join(out_dir, "silhouette.csv"))
        print("\nSilhouette by number of roles (higher is better):")
        print(scores.round(3).to_string())
        silhouette_k = int(scores.idxmax())
        print(f"Best silhouette: {silhouette_k} roles")
        best_k = jump_k if args.select == "jump" else silhouette_k
    print(f"Chosen: {best_k} roles")

    figure, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})
    cut = (tree[-best_k, 2] + tree[-best_k + 1, 2]) / 2
    dendrogram(tree, truncate_mode="lastp", p=40, color_threshold=cut, ax=axes[0],
               leaf_rotation=90, show_contracted=True)
    axes[0].axhline(cut, color="grey", linestyle="--")
    axes[0].set_title(f"Ward dendrogram (top 40 groups), cut into {best_k} roles")
    axes[0].set_ylabel("Merge distance")
    shown = range(1, args.k_max + 1)
    axes[1].plot(list(shown), heights[: len(shown)], marker="o")
    axes[1].axvline(best_k, color="grey", linestyle="--")
    axes[1].set_xlabel("Number of roles")
    axes[1].set_ylabel("Height of merge")
    axes[1].set_title("Merge distance by number of roles (look for the big drops)")
    figure.tight_layout()
    figure.savefig(os.path.join(out_dir, "dendrogram.png"), dpi=150)
    plt.close(figure)

    labels = fcluster(tree, best_k, criterion="maxclust") - 1
    printSplits(tree, z, best_k)
    seasonFeatures.describeClusters(info, z, labels, space, out_dir)
    seasonFeatures.compareToListedPositions(info, labels, out_dir)
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
