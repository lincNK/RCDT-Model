#!/usr/bin/env python3
"""Combine supplementary figures into a single PDF for bioRxiv submission."""

import os
from matplotlib.image import imread
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "figs")
OUTPUT = os.path.join(os.path.dirname(__file__), "RCDT_supplementary.pdf")

FILES = [
    ("fig2_supp_shuffled_control.png", "Receptor Shuffling Control"),
    ("fig2_supp_bifurcation_sweep.png", "Bifurcation Sweep"),
]

def main():
    paths = [os.path.join(FIG_DIR, f) for f, _ in FILES]
    titles = [t for _, t in FILES]

    fig, axes = plt.subplots(len(paths), 1, figsize=(8, 5 * len(paths)))
    if len(paths) == 1:
        axes = [axes]
    for ax, path, title in zip(axes, paths, titles):
        ax.imshow(imread(path))
        ax.axis("off")
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT}")

if __name__ == "__main__":
    main()
