# plot_erac_objective_vs_iter_for_H.py
# ------------------------------------------------------------
# For a chosen environment (same CLI choices as training script),
# plot the exact objective J_exact(k) vs iteration k for multiple H values.
#
# Assumptions about saved files:
#   Training script saved results under:
#     outdir/<tag>/run_<r>_results.pkl
#   Each pickle contains:
#     run_data["history"]["J_exact"]  shape (K,)
#
# Example:
#   python plot_erac_objective_vs_iter_for_H.py \
#       --env random_mdp --S 20 --A 5 --K 2000 --Hs 5,10,50 \
#       --eta_c 0.1 --eta_a 0.05 --lam 0.05 --gamma 0.99 \
#       --runs 4 --outdir ./experiments/ERAC --savedir ./plots/ERAC
#
#   python plot_erac_objective_vs_iter_for_H.py \
#       --env gridworld --rows 4 --cols 4 --K 3000 --Hs 5,20,50 \
#       --eta_c 0.2 --eta_a 0.05 --lam 0.02 --gamma 0.99 \
#       --runs 4 --outdir ./experiments/ERAC --savedir ./plots/ERAC
# ------------------------------------------------------------

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


# -----------------------------
# IO
# -----------------------------
def load_results_from_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


# -----------------------------
# Folder tag MUST match training script
# -----------------------------
def make_tag(args, S, A, H):
    # Must mirror the training script tag:
    # tag = f"{env_name}_S{S}_A{A}_K{K}_H{H}_etaC{eta_c}_etaA{eta_a}_lam{lam}_g{gamma}"
    return f"{args.env}_S{S}_A{A}_K{args.K}_H{H}_etaC{args.eta_c}_etaA{args.eta_a}_lam{args.lam}_g{args.gamma}"


def env_dims_from_args(args):
    if args.env == "random_mdp":
        return args.S, args.A
    if args.env == "gridworld":
        return args.rows * args.cols, 4
    raise ValueError(f"Unknown env {args.env}")


# -----------------------------
# Main plot routine
# -----------------------------
def collect_histories_for_H(args, H, runs, S, A):
    tag = make_tag(args, S, A, H)
    folder = os.path.join(args.outdir, tag)

    # Load runs
    Js = []
    for run in range(runs):
        pkl_path = os.path.join(folder, f"run_{run}_results.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"Missing file: {pkl_path}\n"
                f"Check that your training script used the same outdir/tag and runs indexing."
            )
        data = load_results_from_pickle(pkl_path)
        J = np.asarray(data["history"]["J_exact"], dtype=np.float64)
        if len(J) != args.K:
            # allow mismatch if user changed K; crop to min
            Kmin = min(len(J), args.K)
            J = J[:Kmin]
        Js.append(J)

    Js = np.stack(Js, axis=0)  # (runs, K)
    mean = Js.mean(axis=0)
    std = Js.std(axis=0)
    return mean, std


def main():
    parser = argparse.ArgumentParser("Plot ER-AC exact objective vs iterations for multiple H")

    # --- Environment choice (same as training)
    parser.add_argument("--env", type=str, default="random_mdp", choices=["random_mdp", "gridworld"])

    # random_mdp params
    parser.add_argument("--S", type=int, default=10)
    parser.add_argument("--A", type=int, default=4)

    # gridworld params (only needed for tag consistency)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)

    # --- Hyperparams used in tag (must match training!)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--eta_c", type=float, default=0.1)
    parser.add_argument("--eta_a", type=float, default=0.05)
    parser.add_argument("--K", type=int, default=2000)

    # --- H sweep
    parser.add_argument("--Hs", type=str, default="5,10,50",
                        help="Comma-separated list of H values. Example: 5,10,50")

    # --- Runs + dirs
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--outdir", type=str, default="./experiments/ERAC",
                        help="Where training script saved results.")
    parser.add_argument("--savedir", type=str, default="./plots/ERAC",
                        help="Where to save the pdf plot.")
    parser.add_argument("--filename", type=str, default="objective_vs_iter.pdf",
                        help="Name of the PDF output file.")
    parser.add_argument("--logy", action="store_true", help="Use log scale on y-axis.")
    parser.add_argument("--logx", action="store_true", help="Use log scale on x-axis.")

    args = parser.parse_args()

    H_list = [int(x.strip()) for x in args.Hs.split(",") if x.strip() != ""]
    if len(H_list) == 0:
        raise ValueError("Provide at least one H in --Hs, e.g. --Hs 5,10,50")

    S, A = env_dims_from_args(args)

    create_folder_if_not_exists(args.savedir)

    # Plot style
    font = {"family": "serif"}
    colors = sns.color_palette("colorblind", n_colors=max(3, len(H_list)))
    markers = ["o", "s", "D", "^", "v", "x", "+", "p", "h", "<", ">"]

    fig = plt.figure(figsize=(4.5, 3.3))

    legend_elements = []
    for i, H in enumerate(H_list):
        mean, std = collect_histories_for_H(args, H, args.runs, S, A)

        x = np.arange(len(mean))

        plt.plot(
            x,
            mean,
            color=colors[i],
            marker=markers[i % len(markers)],
            markevery=max(1, len(x) // 8),
            markersize=6,
            linewidth=1.2,
        )
        plt.fill_between(
            x,
            mean - std,
            mean + std,
            color=colors[i],
            alpha=0.25,
            linewidth=0.0,
        )
        legend_elements.append(
            Line2D([0], [0], color=colors[i], marker=markers[i % len(markers)],
                   label=rf"$H={H}$", linewidth=1.2)
        )

    fontsize = 15
    plt.xlabel("Iteration $k$", fontsize=fontsize, **font)
    plt.ylabel(r"Exact objective $J_\lambda(\theta_k)$", fontsize=fontsize, **font)

    if args.logx:
        plt.xscale("log")
    if args.logy:
        plt.yscale("log")

    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)

    plt.legend(handles=legend_elements, fontsize=12, loc="best")
    plt.grid(linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Save
    env_label = args.env
    out_pdf = os.path.join(args.savedir, f"{env_label}_{args.filename}")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved plot to: {out_pdf}")


if __name__ == "__main__":
    main()
