#!/usr/bin/env python3

# Positional roles with k-means, the number of roles chosen by the gap statistic, the silhouette
# score or the elbow in within-cluster spread.
#
# What it does: splits the player-seasons into k groups so that each player is as close as
# possible to the centre of their group, for every k in a range. Three measures of how many
# groups the data supports are computed for each k:
#   - inertia (total squared distance to the group centres): always falls as k rises; the
#     "elbow" where it stops falling quickly is a rough guide
#   - silhouette score: how much closer players are to their own group than to the next nearest
#     (-1 to 1, higher is better)
#   - gap statistic: how much tighter the groups are than groups found in random data with no
#     structure; the chosen k is the smallest whose gap is within one standard error of the next
# The chosen k (by the gap statistic unless --select says otherwise) is then fitted and reported.
#
# Why it matters for the analysis: k-means is the simplest, fastest and most widely used
# clustering method, which makes it the baseline the other methods should beat. It assumes round
# groups of similar size, which roles aren't (there are far fewer rucks than midfielders), so
# where it disagrees with the Gaussian mixture shows which roles depend on that assumption.
# Its hard labels are also easy to use directly as a categorical feature.
#
# Usage: ../venv/bin/python kmeansClustering.py --years 2021-2025 [--select silhouette | --k 10]

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import seasonFeatures


def gapStatistic(space, k, references, seed):
    """Tibshirani's gap statistic for k groups: log inertia of `references` random datasets
    (uniform within the bounding box of the data's principal axes) minus the data's log inertia.
    Returns (gap, standard error)."""
    rng = np.random.default_rng(seed)
    centred = space - space.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    rotated = centred @ axes.T
    low, high = rotated.min(axis=0), rotated.max(axis=0)

    actual = np.log(KMeans(n_clusters=k, n_init=5, random_state=seed).fit(space).inertia_)
    reference = np.array([
        np.log(KMeans(n_clusters=k, n_init=3, random_state=seed)
               .fit(rng.uniform(low, high, size=rotated.shape) @ axes).inertia_)
        for _ in range(references)
    ])
    return reference.mean() - actual, reference.std() * np.sqrt(1 + 1 / references)


def main():
    parser = seasonFeatures.newParser("Positional roles with k-means")
    parser.add_argument("--k-min", type=int, default=4, help="Fewest roles to try (default 4)")
    parser.add_argument("--k-max", type=int, default=16, help="Most roles to try (default 16)")
    parser.add_argument("--k", type=int, help="Use this many roles instead of choosing")
    parser.add_argument(
        "--select", choices=["gap", "silhouette"], default="gap",
        help="How to choose the number of roles (default gap)",
    )
    parser.add_argument(
        "--references", type=int, default=10,
        help="Random reference datasets per k for the gap statistic (default 10)",
    )
    args = parser.parse_args()

    info, z = seasonFeatures.prepare(args)
    space = seasonFeatures.reduce(z, args.pca_variance, args.seed)
    out_dir = seasonFeatures.outputDirectory("kmeans", args)

    if args.k:
        best_k = args.k
    else:
        ks = list(range(args.k_min, args.k_max + 2))  # one past k_max for the gap rule
        scores = pd.DataFrame(index=ks, columns=["inertia", "silhouette", "gap", "gap_se"],
                              dtype=float)
        scores.index.name = "k"
        for k in ks:
            model = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit(space)
            scores.loc[k, "inertia"] = model.inertia_
            scores.loc[k, "silhouette"] = silhouette_score(space, model.labels_)
            scores.loc[k, "gap"], scores.loc[k, "gap_se"] = gapStatistic(
                space, k, args.references, args.seed)
            print(f"  k={k}: silhouette {scores.loc[k, 'silhouette']:.3f}, "
                  f"gap {scores.loc[k, 'gap']:.3f}")
        scores.to_csv(os.path.join(out_dir, "model_selection.csv"))

        # Gap rule: smallest k with gap(k) >= gap(k+1) - se(k+1); k_max if none qualifies
        gap_k = next(
            (k for k in ks[:-1]
             if scores.loc[k, "gap"] >= scores.loc[k + 1, "gap"] - scores.loc[k + 1, "gap_se"]),
            None,
        )
        if gap_k is None:
            gap_k = args.k_max
            print("\nThe gap statistic kept rising across the whole range, so it has no clear "
                  "answer (common when groups blend into each other); using --k-max")
        silhouette_k = int(scores.loc[: args.k_max, "silhouette"].idxmax())
        print(f"\nGap statistic chooses {gap_k} roles; best silhouette is {silhouette_k} roles")
        best_k = gap_k if args.select == "gap" else silhouette_k

        shown = scores.loc[: args.k_max]
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].plot(shown.index, shown["inertia"], marker="o")
        axes[0].set_title("Inertia (look for the elbow)")
        axes[1].plot(shown.index, shown["silhouette"], marker="o")
        axes[1].set_title("Silhouette (higher is better)")
        axes[2].errorbar(shown.index, shown["gap"], yerr=shown["gap_se"], marker="o", capsize=3)
        axes[2].set_title("Gap statistic")
        for axis in axes:
            axis.axvline(best_k, color="grey", linestyle="--")
            axis.set_xlabel("Number of roles (k)")
        figure.suptitle(f"k-means model selection, {best_k} roles chosen ({args.select})")
        figure.tight_layout()
        figure.savefig(os.path.join(out_dir, "model_selection.png"), dpi=150)
        plt.close(figure)

    model = KMeans(n_clusters=best_k, n_init=20, random_state=args.seed).fit(space)
    print(f"\nk-means with {best_k} roles, silhouette "
          f"{silhouette_score(space, model.labels_):.3f}")
    seasonFeatures.describeClusters(info, z, model.labels_, space, out_dir)
    seasonFeatures.compareToListedPositions(info, model.labels_, out_dir)
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
