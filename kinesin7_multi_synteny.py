"""
kinesin7_multi_synteny.py
─────────────────────────
Analyse and visualise multiple synteny of Kinesin-7 genes across all
eight inter-/intra-subgenome anchor pairs.

Outputs
  figures/kinesin7_multi_synteny.png   -- heatmap + histogram
  (printed summary table to stdout)
"""

import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

DIR    = r"G:\iub_research\kinesin7_synteny\2026-04-14"
FIGDIR = os.path.join(DIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

KINESIN_FILE = os.path.join(DIR, "2026-04-15_kinesin7_genes.txt")

ANCHOR_PAIRS = [
    ("Ga_GhA",  "Ga_GhA.anchors"),
    ("Ga_GbA",  "Ga_GbA.anchors"),
    ("GhA_GbA", "GhA_GbA.anchors"),
    ("GhA_GhD", "GhA_GhD.anchors"),
    ("GhD_GbD", "GhD_GbD.anchors"),
    ("Gr_GhD",  "Gr_GhD.anchors"),
    ("Gr_GbD",  "Gr_GbD.anchors"),
    ("GbA_GbD", "GbA_GbD.anchors"),
]

# Subgenome colour for gene rows in heatmap
SUBGENOME_COLORS = {
    "GhA": "#e74c3c",
    "GhD": "#c0392b",
    "GbA": "#3498db",
    "GbD": "#2980b9",
    "Ga":  "#2ecc71",
    "Gr":  "#f39c12",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def subgenome_of(gene):
    if gene.startswith("GohirA"): return "GhA"
    if gene.startswith("GohirD"): return "GhD"
    if gene.startswith("GobarA"): return "GbA"
    if gene.startswith("GobarD"): return "GbD"
    if gene.startswith("Gorai"):  return "Gr"
    if gene.startswith("Ga"):     return "Ga"
    return "?"


def load_kinesin_genes(path):
    genes = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                genes.append(ln)
    return genes


def load_anchors(pairs, base_dir):
    """Return gene_appearances: gene -> {pair_name: [partner, ...]}"""
    gene_appearances = defaultdict(lambda: defaultdict(list))
    pair_counts = {}
    for pair_name, fname in pairs:
        fpath = os.path.join(base_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [WARN] missing: {fpath}")
            pair_counts[pair_name] = 0
            continue
        n = 0
        with open(fpath) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                g1, g2 = parts[0], parts[1]
                gene_appearances[g1][pair_name].append(g2)
                gene_appearances[g2][pair_name].append(g1)
                n += 1
        pair_counts[pair_name] = n
    return gene_appearances, pair_counts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    kinesin_genes = load_kinesin_genes(KINESIN_FILE)
    print(f"Kinesin-7 genes loaded: {len(kinesin_genes)}")

    gene_appearances, pair_counts = load_anchors(ANCHOR_PAIRS, DIR)
    pair_names = [p for p, _ in ANCHOR_PAIRS]

    # Print pair-level hit counts
    print("\nAnchor-pair kinesin hit counts:")
    for pn in pair_names:
        total_hits = sum(
            1 for g in kinesin_genes
            for _ in gene_appearances[g].get(pn, [])
        )
        print(f"  {pn:<14}  {total_hits:>4} kinesin gene appearances  "
              f"({pair_counts.get(pn, 0)} total anchor lines)")

    # Build presence/absence matrix (genes × pairs)
    # Sort genes by subgenome order
    sg_order = ["GhA", "GhD", "GbA", "GbD", "Ga", "Gr"]
    genes_sorted = sorted(
        kinesin_genes,
        key=lambda g: (sg_order.index(subgenome_of(g))
                       if subgenome_of(g) in sg_order else 99, g)
    )

    matrix = np.zeros((len(genes_sorted), len(pair_names)), dtype=int)
    for i, g in enumerate(genes_sorted):
        for j, pn in enumerate(pair_names):
            if pn in gene_appearances[g]:
                matrix[i, j] = 1

    pair_count_per_gene = matrix.sum(axis=1)   # how many pairs each gene hits
    gene_count_per_pair = matrix.sum(axis=0)   # how many genes per pair

    # Summary
    in_multiple = (pair_count_per_gene > 1).sum()
    in_any      = (pair_count_per_gene > 0).sum()
    in_none     = (pair_count_per_gene == 0).sum()
    print(f"\nGenes in ≥1 pair:     {in_any}/{len(kinesin_genes)}")
    print(f"Genes in >1 pair:     {in_multiple}/{len(kinesin_genes)}  (multiple synteny)")
    print(f"Genes in 0 pairs:     {in_none}/{len(kinesin_genes)}")
    print(f"\nPair-count distribution:")
    for n in range(0, pair_count_per_gene.max() + 1):
        cnt = (pair_count_per_gene == n).sum()
        if cnt:
            bar = "█" * cnt
            print(f"  {n:>2} pairs: {cnt:>3}  {bar}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Multiple Synteny of Kinesin-7 Genes Across Anchor Pairs",
                 fontsize=14, fontweight="bold", y=0.98)

    # Layout: heatmap on top (wider), histogram on bottom-left, bar on bottom-right
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.3], hspace=0.45, wspace=0.35,
                          left=0.13, right=0.97, top=0.93, bottom=0.07)

    ax_heat  = fig.add_subplot(gs[0, :])   # heatmap spans both columns
    ax_hist  = fig.add_subplot(gs[1, 0])   # pairs-per-gene histogram
    ax_bar   = fig.add_subplot(gs[1, 1])   # genes-per-pair bar

    # ── Heatmap ───────────────────────────────────────────────────────────────
    # Colour cells: 1 = present (pair-specific colour), 0 = absent (light grey)
    cmap_base = matplotlib.colormaps.get_cmap("Set2").resampled(len(pair_names))
    rgb_matrix = np.ones((len(genes_sorted), len(pair_names), 3))

    for j in range(len(pair_names)):
        colour = np.array(cmap_base(j)[:3])
        for i in range(len(genes_sorted)):
            if matrix[i, j]:
                rgb_matrix[i, j] = colour
            else:
                rgb_matrix[i, j] = np.array([0.93, 0.93, 0.93])

    ax_heat.imshow(rgb_matrix, aspect="auto", interpolation="nearest")

    # Subgenome row-colour strip on left
    strip_x = -1.5
    prev_sg = None
    sg_label_positions = []
    for i, g in enumerate(genes_sorted):
        sg = subgenome_of(g)
        if sg != prev_sg:
            sg_label_positions.append((i, sg))
            prev_sg = sg
        c = SUBGENOME_COLORS.get(sg, "#aaaaaa")
        ax_heat.add_patch(mpatches.Rectangle(
            (-1.8, i - 0.5), 0.7, 1.0,
            color=c, clip_on=False, transform=ax_heat.transData
        ))

    # Subgenome labels on left strip
    next_positions = [p for p, _ in sg_label_positions[1:]] + [len(genes_sorted)]
    for (start_i, sg), end_i in zip(sg_label_positions, next_positions):
        mid = (start_i + end_i - 1) / 2
        ax_heat.text(-2.5, mid, sg, va="center", ha="right",
                     fontsize=8, fontweight="bold",
                     color=SUBGENOME_COLORS.get(sg, "#555555"))

    # Gene labels on y-axis (show if ≤ 101)
    ax_heat.set_yticks(range(len(genes_sorted)))
    ax_heat.set_yticklabels(genes_sorted, fontsize=5.5)

    ax_heat.set_xticks(range(len(pair_names)))
    ax_heat.set_xticklabels(pair_names, rotation=35, ha="right", fontsize=9)
    ax_heat.set_xlim(-2.0, len(pair_names) - 0.5)
    ax_heat.set_ylim(len(genes_sorted) - 0.5, -0.5)

    # Count column on right: how many pairs each gene hits
    for i, cnt in enumerate(pair_count_per_gene):
        ax_heat.text(len(pair_names) - 0.3, i, str(cnt),
                     va="center", ha="left", fontsize=5.5,
                     color="#333333")
    ax_heat.text(len(pair_names) - 0.3, -1.2, "n", va="center",
                 ha="left", fontsize=7, color="#555555", style="italic")

    # Gene count per pair at bottom
    for j, cnt in enumerate(gene_count_per_pair):
        ax_heat.text(j, len(genes_sorted) + 0.4, str(cnt),
                     va="top", ha="center", fontsize=7, color="#333333")

    ax_heat.set_title("Presence/absence across anchor pairs  (rows = genes, cols = pairs)",
                      fontsize=10, pad=6)

    # Legend patches for pair colours
    legend_patches = [
        mpatches.Patch(color=cmap_base(j)[:3], label=pair_names[j])
        for j in range(len(pair_names))
    ]
    ax_heat.legend(handles=legend_patches, loc="upper right",
                   fontsize=7, ncol=2, title="Pair", title_fontsize=7,
                   bbox_to_anchor=(1.0, 1.18))

    # ── Histogram: distribution of pair counts per gene ───────────────────────
    max_pairs = int(pair_count_per_gene.max())
    bins = np.arange(-0.5, max_pairs + 1.5)
    counts, _ = np.histogram(pair_count_per_gene, bins=bins)
    bin_centers = np.arange(0, max_pairs + 1)

    bar_colors = ["#d5d5d5" if b == 0 else "#3498db" for b in bin_centers]
    ax_hist.bar(bin_centers, counts, color=bar_colors, edgecolor="white", width=0.7)
    for x, y in zip(bin_centers, counts):
        if y > 0:
            ax_hist.text(x, y + 0.3, str(y), ha="center", va="bottom", fontsize=8)
    ax_hist.set_xlabel("Number of anchor pairs gene appears in", fontsize=9)
    ax_hist.set_ylabel("Gene count", fontsize=9)
    ax_hist.set_title("Pairs-per-gene distribution", fontsize=10)
    ax_hist.set_xticks(bin_centers)
    ax_hist.set_xlim(-0.7, max_pairs + 0.7)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    # ── Bar: genes per pair ───────────────────────────────────────────────────
    bar_cols = [cmap_base(j)[:3] for j in range(len(pair_names))]
    ax_bar.bar(range(len(pair_names)), gene_count_per_pair, color=bar_cols,
               edgecolor="white")
    for j, cnt in enumerate(gene_count_per_pair):
        ax_bar.text(j, cnt + 0.3, str(cnt), ha="center", va="bottom", fontsize=8)
    ax_bar.set_xticks(range(len(pair_names)))
    ax_bar.set_xticklabels(pair_names, rotation=35, ha="right", fontsize=8)
    ax_bar.set_ylabel("Kinesin-7 genes with syntenic hit", fontsize=9)
    ax_bar.set_title("Syntenic Kinesin-7 genes per pair", fontsize=10)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(FIGDIR, "kinesin7_multi_synteny.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


if __name__ == "__main__":
    main()
