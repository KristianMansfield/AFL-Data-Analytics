#!/usr/bin/env python3

# Positional roles with a Gaussian Mixture Model (GMM), the number of roles chosen by BIC.
#
# What it does: models the player-seasons as a mixture of overlapping bell-shaped groups, one per
# role. It fits mixtures for every number of roles in a range (and several shapes of group), scores
# each with the Bayesian Information Criterion (BIC), which rewards fitting the data well but
# penalises every extra parameter, and keeps the mixture with the lowest BIC. Each player then
# gets a probability of belonging to each role, not just one label.
#
# Why it matters for the analysis: BIC gives a principled answer to "how many roles are there?"
# rather than a guess, and the probabilities capture hybrid players (a mid-forward might be 60%
# inside midfielder and 40% small forward). Those probabilities are useful features in their own
# right when predicting awards, and the players with no clear role are worth looking at
# separately. This is the recommended primary method; the other scripts are checks on it.
#
# Usage: ../venv/bin/python gmmClustering.py --years 2025 [--k-min 4 --k-max 16 | --k 10]

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

import seasonFeatures

COVARIANCE_TYPES = ["full", "tied", "diag", "spherical"]


def writeRoles(info, z, labels, probabilities, hybrid_threshold, out_dir):
    """Write every player's name and team, grouped by role, to roles.txt: each role headed by
    its most distinctive stats, players listed from most to least certain. Hybrid players
    (most likely role below the threshold) are listed at the end with their two likeliest roles."""
    path = os.path.join(out_dir, "roles.txt")
    profiles = z.groupby(labels).mean()
    pooled = info.index.get_level_values("year").nunique() > 1
    confidence = probabilities.max(axis=1)

    with open(path, "w", encoding="utf-8") as fh:
        for cluster in sorted(set(labels)):
            members = np.flatnonzero(labels == cluster)
            profile = profiles.loc[cluster].sort_values(ascending=False)
            fh.write(f"=== Cluster {cluster}: {len(members)} player-seasons ===\n")
            fh.write("High: " + ", ".join(f"{c} {v:+.1f}" for c, v in profile[:6].items())
                     + "\n")
            fh.write("Low:  " + ", ".join(f"{c} {v:+.1f}" for c, v in profile[::-1][:3].items())
                     + "\n\n")
            fh.write(f"{'Player':<28} {'Team':<6} {'Year' if pooled else '':<5} "
                     f"{'Games':>5} {'Prob':>5}\n")
            for i in members[np.argsort(-confidence[members])]:
                (year, _), row = info.index[i], info.iloc[i]
                fh.write(f"{row['name']:<28} {row['team']:<6} {year if pooled else '':<5} "
                         f"{row['games']:>5} {confidence[i]:>5.0%}\n")
            fh.write("\n")

        hybrids = np.flatnonzero(confidence < hybrid_threshold)
        fh.write(f"=== Hybrids: {len(hybrids)} player-seasons with their most likely role "
                 f"below {hybrid_threshold:.0%} ===\n")
        for i in hybrids[np.argsort(confidence[hybrids])]:
            top_two = np.argsort(probabilities[i])[::-1][:2]
            split = ", ".join(f"cluster {c} {probabilities[i][c]:.0%}" for c in top_two)
            fh.write(f"{seasonFeatures.playerLabel(info, info.index[i])}: {split}\n")
    return path


def main():
    parser = seasonFeatures.newParser("Positional roles with a Gaussian Mixture Model, k chosen by BIC")
    parser.add_argument("--k-min", type=int, default=4, help="Fewest roles to try (default 4)")
    parser.add_argument("--k-max", type=int, default=16, help="Most roles to try (default 16)")
    parser.add_argument("--k", type=int, help="Use this many roles instead of choosing by BIC")
    parser.add_argument(
        "--covariance", choices=COVARIANCE_TYPES,
        help="Only try this group shape (default: try them all and keep the best BIC)",
    )
    parser.add_argument(
        "--hybrid-threshold", type=float, default=0.7,
        help="Report players whose most likely role has a probability below this (default 0.7)",
    )
    args = parser.parse_args()

    info, z = seasonFeatures.prepare(args)
    space = seasonFeatures.reduce(z, args.pca_variance, args.seed)
    out_dir = seasonFeatures.outputDirectory("gmm", args)

    covariances = [args.covariance] if args.covariance else COVARIANCE_TYPES
    ks = [args.k] if args.k else list(range(args.k_min, args.k_max + 1))

    # BIC for every number of roles and group shape
    bic = pd.DataFrame(index=ks, columns=covariances, dtype=float)
    models = {}
    for covariance in covariances:
        for k in ks:
            model = GaussianMixture(n_components=k, covariance_type=covariance, n_init=5,
                                    random_state=args.seed).fit(space)
            bic.loc[k, covariance] = model.bic(space)
            models[k, covariance] = model
    bic.index.name = "k"
    bic.to_csv(os.path.join(out_dir, "bic.csv"))
    print("\nBIC (lower is better):")
    print(bic.round(0).to_string())

    best_k, best_covariance = bic.stack().idxmin()
    model = models[best_k, best_covariance]
    print(f"\nBest: {best_k} roles with '{best_covariance}' covariance")

    if len(ks) > 1:
        plt.figure(figsize=(9, 5))
        for covariance in covariances:
            plt.plot(bic.index, bic[covariance], marker="o", label=covariance)
        plt.axvline(best_k, color="grey", linestyle="--")
        plt.xlabel("Number of roles (k)")
        plt.ylabel("BIC (lower is better)")
        plt.title(f"GMM model selection, {best_k} roles chosen")
        plt.legend(title="covariance")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "bic.png"), dpi=150)
        plt.close()

    labels = model.predict(space)
    probabilities = model.predict_proba(space)

    seasonFeatures.describeClusters(info, z, labels, space, out_dir)
    seasonFeatures.compareToListedPositions(info, labels, out_dir)

    membership = info[["name", "team", "games", "listed_position"]].copy()
    membership["cluster"] = labels
    membership["confidence"] = probabilities.max(axis=1)
    for cluster in range(best_k):
        membership[f"p_cluster_{cluster}"] = probabilities[:, cluster]
    membership.to_csv(os.path.join(out_dir, "probabilities.csv"))

    roles_file = writeRoles(info, z, labels, probabilities, args.hybrid_threshold, out_dir)
    hybrids = (membership["confidence"] < args.hybrid_threshold).sum()
    print(f"\nEvery player in each role ({hybrids} hybrids with their most likely role below "
          f"{args.hybrid_threshold:.0%}) written to {roles_file}")

    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
