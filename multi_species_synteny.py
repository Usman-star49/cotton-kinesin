"""
multi_species_synteny.py
────────────────────────
Four-section multi-species kinesin-7 synteny analysis.

Section 1 – Comparative synteny matrix
    Conservation counts for every pairwise combination of the 6 subgenomes.
    Output: figures/mss_1_comparative_matrix.png

Section 2 – Collinearity analysis
    Chromosome-ordered dot-strip for each anchor pair showing block structure
    of kinesin genes relative to all collinear gene pairs.
    Output: figures/mss_2_collinearity.png

Section 3 – Inter-species dot-plots
    Classic synteny dot-plot (gene rank × gene rank) for every anchor pair
    that involves at least one kinesin hit.
    Output: figures/mss_3_dotplots.png

Section 4 – Multi-species network
    Nodes = kinesin genes (coloured by subgenome), edges = syntenic anchors.
    Node size ∝ degree; layout = spring / force-directed.
    Output: figures/mss_4_network.png

All outputs are written to:
    G:/iub_research/kinesin7_synteny/2026-04-14/figures/
"""

import os
import re
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DIR     = r"G:\iub_research\kinesin7_synteny\2026-04-14"
FIGDIR  = os.path.join(DIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

KINESIN_FILE = os.path.join(DIR, "2026-04-15_kinesin7_genes.txt")

# All anchor pairs (query, subject, filename)
ANCHOR_PAIRS = [
    ("Ga",  "GhA", "Ga_GhA.anchors"),
    ("Ga",  "GbA", "Ga_GbA.anchors"),
    ("GhA", "GbA", "GhA_GbA.anchors"),
    ("GhA", "GhD", "GhA_GhD.anchors"),
    ("Gr",  "GhD", "Gr_GhD.anchors"),
    ("Gr",  "GbD", "Gr_GbD.anchors"),
    ("GhD", "GbD", "GhD_GbD.anchors"),
    ("GbA", "GbD", "GbA_GbD.anchors"),
]

SUBGENOMES = ["Ga", "Gr", "GhA", "GhD", "GbA", "GbD"]

SP_COLOR = {
    "Ga":  "#9b59b6",
    "Gr":  "#e67e22",
    "GhA": "#2980b9",
    "GhD": "#1a5276",
    "GbA": "#27ae60",
    "GbD": "#1e8449",
}

SP_LABEL = {
    "Ga":  "G. arboreum (Ga)",
    "Gr":  "G. raimondii (Gr)",
    "GhA": "G. hirsutum A (GhA)",
    "GhD": "G. hirsutum D (GhD)",
    "GbA": "G. barbadense A (GbA)",
    "GbD": "G. barbadense D (GbD)",
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def sp_of(gene):
    if gene.startswith("GohirA"): return "GhA"
    if gene.startswith("GohirD"): return "GhD"
    if gene.startswith("GobarA"): return "GbA"
    if gene.startswith("GobarD"): return "GbD"
    if gene.startswith("Gorai"):  return "Gr"
    if gene.startswith("Ga"):     return "Ga"
    return None


def load_kinesin(path):
    genes = set()
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                genes.add(ln)
    return genes


def load_bed(sp):
    """Return {gene: (chr, start, rank)}, chr_order."""
    path = os.path.join(DIR, sp + ".bed")
    rows = []
    with open(path) as fh:
        for ln in fh:
            p = ln.split()
            if len(p) >= 4:
                rows.append((p[0], int(p[1]), p[3]))   # chr, start, gene
    def _key(r):
        m = re.search(r'(\d+)$', r[0])
        return (int(m.group(1)) if m else 0, r[1])
    rows.sort(key=_key)
    gene_info = {}
    chr_order = []
    seen_chr  = set()
    for rank, (ch, st, gid) in enumerate(rows):
        gene_info[gid] = (ch, st, rank)
        if ch not in seen_chr:
            chr_order.append(ch)
            seen_chr.add(ch)
    return gene_info, chr_order


def load_anchors(fname):
    """Return list of (g1, g2, score)."""
    path = os.path.join(DIR, fname)
    pairs = []
    if not os.path.exists(path):
        return pairs
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            p = ln.split()
            if len(p) >= 2:
                score = int(p[2]) if len(p) >= 3 else 1
                pairs.append((p[0], p[1], score))
    return pairs


# Pre-load everything
kinesin_genes = load_kinesin(KINESIN_FILE)

bed = {}
chr_order = {}
for sp in SUBGENOMES:
    bed[sp], chr_order[sp] = load_bed(sp)

anchors = {}   # (sp1, sp2) -> [(g1, g2, score)]
for sp1, sp2, fname in ANCHOR_PAIRS:
    anchors[(sp1, sp2)] = load_anchors(fname)
    anchors[(sp2, sp1)] = [(g2, g1, s) for g1, g2, s in anchors[(sp1, sp2)]]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1  –  COMPARATIVE SYNTENY MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def section1_comparative_matrix():
    """
    6×6 heatmap: cell (i,j) = number of kinesin genes from subgenome i that
    have a syntenic anchor partner in subgenome j.
    Upper triangle: count of syntenic kinesin GENE PAIRS.
    Diagonal: total kinesin genes in that subgenome.
    """
    print("\n[Section 1] Comparative synteny matrix …")

    n = len(SUBGENOMES)
    matrix = np.zeros((n, n), dtype=int)

    # Diagonal = gene count
    for i, sp in enumerate(SUBGENOMES):
        matrix[i, i] = sum(1 for g in kinesin_genes if sp_of(g) == sp)

    # Off-diagonal: count unique kinesin pairs per ordered pair
    pair_set = {}  # (i, j) -> set of (g1, g2) kinesin pairs
    for i, sp1 in enumerate(SUBGENOMES):
        for j, sp2 in enumerate(SUBGENOMES):
            if i == j:
                continue
            key = (sp1, sp2)
            if key not in anchors:
                continue
            cnt = 0
            for g1, g2, _ in anchors[key]:
                if g1 in kinesin_genes and g2 in kinesin_genes:
                    cnt += 1
            matrix[i, j] = cnt

    # Symmetrise upper/lower for display (use max)
    for i in range(n):
        for j in range(i + 1, n):
            val = max(matrix[i, j], matrix[j, i])
            matrix[i, j] = val
            matrix[j, i] = val

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5),
                             gridspec_kw={"width_ratios": [1, 1.05]})
    fig.suptitle("Kinesin-7 Comparative Synteny Across Species",
                 fontsize=13, fontweight="bold")

    # ── Left: heatmap ─────────────────────────────────────────────────────────
    ax = axes[0]
    diag_mask = np.eye(n, dtype=bool)
    off_vals  = matrix[~diag_mask].astype(float)
    vmax      = off_vals.max() if off_vals.max() > 0 else 1

    display = matrix.astype(float)
    display[diag_mask] = np.nan

    im = ax.imshow(display, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")

    # Diagonal cells: grey with gene count
    for i in range(n):
        ax.add_patch(mpatches.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                        facecolor="#cccccc", zorder=2))
        ax.text(i, i, str(matrix[i, i]),
                ha="center", va="center", fontsize=10,
                fontweight="bold", zorder=3, color="#333333")

    # Annotate off-diagonal
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = matrix[i, j]
            tc  = "white" if val > vmax * 0.6 else "#333333"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=9, color=tc)

    labels = [SP_LABEL[s] for s in SUBGENOMES]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Syntenic kinesin gene pairs\n(diagonal = gene count per subgenome)",
                 fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Syntenic pairs")

    # Add coloured border to tick labels
    for i, (xtl, ytl) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels())):
        col = SP_COLOR[SUBGENOMES[i]]
        xtl.set_color(col)
        ytl.set_color(col)

    # ── Right: stacked bar — for each subgenome, how many kinesin genes
    #    have syntelogs in 0, 1, 2, … other subgenomes ──────────────────────────
    ax2 = axes[1]
    partner_counts = {sp: [] for sp in SUBGENOMES}
    for sp in SUBGENOMES:
        sp_genes = [g for g in kinesin_genes if sp_of(g) == sp]
        for g in sp_genes:
            n_partners = 0
            for sp2 in SUBGENOMES:
                if sp2 == sp:
                    continue
                key = (sp, sp2)
                if key not in anchors:
                    continue
                has = any((g1 == g or g2 == g) and
                          (g2 in kinesin_genes or g1 in kinesin_genes)
                          for g1, g2, _ in anchors[key])
                if has:
                    n_partners += 1
            partner_counts[sp].append(n_partners)

    max_partners = max(max(v) for v in partner_counts.values() if v)
    bar_data = np.zeros((len(SUBGENOMES), max_partners + 1), dtype=int)
    for i, sp in enumerate(SUBGENOMES):
        for v in partner_counts[sp]:
            bar_data[i, v] += 1

    x = np.arange(len(SUBGENOMES))
    bottoms = np.zeros(len(SUBGENOMES))
    cmap_p = plt.get_cmap("Blues")
    for k in range(max_partners + 1):
        color = cmap_p(0.15 + 0.75 * k / max(max_partners, 1))
        ax2.bar(x, bar_data[:, k], bottom=bottoms, color=color,
                edgecolor="white", linewidth=0.5,
                label=f"{k} partner species" if k <= 3 else "_nolegend_")
        for xi, (val, bot) in enumerate(zip(bar_data[:, k], bottoms)):
            if val > 0:
                ax2.text(xi, bot + val / 2, str(val),
                         ha="center", va="center", fontsize=7.5,
                         color="white" if k > 1 else "#333333")
        bottoms += bar_data[:, k]

    ax2.set_xticks(x)
    ax2.set_xticklabels([SP_LABEL[s] for s in SUBGENOMES],
                        rotation=35, ha="right", fontsize=8)
    for i, xtl in enumerate(ax2.get_xticklabels()):
        xtl.set_color(SP_COLOR[SUBGENOMES[i]])
    ax2.set_ylabel("Number of kinesin-7 genes", fontsize=9)
    ax2.set_title("Kinesin genes by number of species with syntelog",
                  fontsize=9)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_ylim(0, max(bottoms) * 1.15)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIGDIR, "mss_1_comparative_matrix.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved →", out)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2  –  COLLINEARITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def section2_collinearity():
    """
    For each anchor pair, draw a collinearity strip:
      x-axis = gene rank in species 1  (all genes, grey)
      y-axis = gene rank in species 2  (all genes, grey)
      Background: thin grey lines connecting all syntenic pairs (collinearity)
      Foreground: highlighted lines for kinesin gene pairs (coloured)
    Eight subplots — one per anchor pair.
    """
    print("\n[Section 2] Collinearity analysis …")

    n_pairs = len(ANCHOR_PAIRS)
    ncols   = 4
    nrows   = (n_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4.5),
                             squeeze=False)
    fig.suptitle("Kinesin-7 Collinearity Analysis — All Anchor Pairs",
                 fontsize=13, fontweight="bold")

    for idx, (sp1, sp2, fname) in enumerate(ANCHOR_PAIRS):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        pairs_all = anchors.get((sp1, sp2), [])
        if not pairs_all:
            ax.set_visible(False)
            continue

        # Gene ranks
        rank1 = {g: info[2] for g, info in bed[sp1].items()}
        rank2 = {g: info[2] for g, info in bed[sp2].items()}
        total1 = len(rank1)
        total2 = len(rank2)

        # Background: all syntenic pairs (random sample for speed)
        import random
        random.seed(42)
        bg = pairs_all if len(pairs_all) <= 5000 else random.sample(pairs_all, 5000)
        xs, ys = [], []
        for g1, g2, _ in bg:
            r1 = rank1.get(g1)
            r2 = rank2.get(g2)
            if r1 is not None and r2 is not None:
                xs.append(r1)
                ys.append(r2)
        ax.scatter(xs, ys, s=0.3, c="#cccccc", linewidths=0, zorder=1, rasterized=True)

        # Chromosome boundaries
        chr_breaks1 = _chr_breaks(sp1)
        chr_breaks2 = _chr_breaks(sp2)
        for brk in chr_breaks1:
            ax.axvline(brk, color="#dddddd", lw=0.4, zorder=1)
        for brk in chr_breaks2:
            ax.axhline(brk, color="#dddddd", lw=0.4, zorder=1)

        # Kinesin pairs
        kx, ky = [], []
        for g1, g2, _ in pairs_all:
            r1 = rank1.get(g1)
            r2 = rank2.get(g2)
            if r1 is None or r2 is None:
                continue
            if g1 in kinesin_genes or g2 in kinesin_genes:
                kx.append(r1)
                ky.append(r2)

        both_kin_x, both_kin_y = [], []
        one_kin_x,  one_kin_y  = [], []
        for g1, g2, _ in pairs_all:
            r1 = rank1.get(g1)
            r2 = rank2.get(g2)
            if r1 is None or r2 is None:
                continue
            b1 = g1 in kinesin_genes
            b2 = g2 in kinesin_genes
            if b1 and b2:
                both_kin_x.append(r1)
                both_kin_y.append(r2)
            elif b1 or b2:
                one_kin_x.append(r1)
                one_kin_y.append(r2)

        ax.scatter(one_kin_x,  one_kin_y,  s=18, c="#f39c12",
                   linewidths=0.4, edgecolors="#c0392b", zorder=3,
                   label="One kinesin")
        ax.scatter(both_kin_x, both_kin_y, s=35, c="#e74c3c",
                   linewidths=0.6, edgecolors="#922b21", zorder=4,
                   label="Both kinesin")

        ax.set_xlim(0, total1)
        ax.set_ylim(0, total2)
        ax.set_xlabel(SP_LABEL[sp1], fontsize=7, color=SP_COLOR[sp1])
        ax.set_ylabel(SP_LABEL[sp2], fontsize=7, color=SP_COLOR[sp2])
        ax.set_title(f"{sp1} vs {sp2}  (kin pairs: {len(both_kin_x)})",
                     fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=6)
        if both_kin_x or one_kin_x:
            ax.legend(fontsize=6, markerscale=0.8, loc="upper left")

    # Hide unused axes
    for idx in range(n_pairs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIGDIR, "mss_2_collinearity.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved →", out)


def _chr_breaks(sp):
    """Return gene-rank positions where chromosome changes (boundary lines)."""
    breaks = []
    prev_ch = None
    for g, (ch, st, rank) in sorted(bed[sp].items(), key=lambda x: x[1][2]):
        if prev_ch and ch != prev_ch:
            breaks.append(rank)
        prev_ch = ch
    return breaks


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3  –  INTER-SPECIES DOT-PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def section3_dotplots():
    """
    For every anchor pair: chromosome-level dot-plot using mid-point bp
    coordinates.  x-axis = sp1 chromosomes, y-axis = sp2 chromosomes.
    Background grey dots = all syntenic pairs; coloured dots = kinesin pairs.
    """
    print("\n[Section 3] Inter-species dot-plots …")

    n_pairs = len(ANCHOR_PAIRS)
    ncols   = 4
    nrows   = (n_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4.5),
                             squeeze=False)
    fig.suptitle("Kinesin-7 Inter-species Synteny Dot-plots",
                 fontsize=13, fontweight="bold")

    for idx, (sp1, sp2, fname) in enumerate(ANCHOR_PAIRS):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        pairs_all = anchors.get((sp1, sp2), [])
        if not pairs_all:
            ax.set_visible(False)
            continue

        # Build per-chromosome cumulative offset (in Mb)
        def _cum_offset(sp):
            """Returns {chr: offset_Mb}, chr_order, total_Mb, boundaries."""
            co  = chr_order[sp]
            # max position per chr
            chr_max = defaultdict(int)
            for g, (ch, st, rk) in bed[sp].items():
                if st > chr_max[ch]:
                    chr_max[ch] = st
            gap_Mb = 5  # visual gap between chromosomes
            offset = {}
            cur = 0.0
            boundaries = []
            for ch in co:
                offset[ch] = cur / 1e6
                cur += chr_max[ch] + gap_Mb * 1e6
                boundaries.append(cur / 1e6)
            return offset, co, cur / 1e6, boundaries

        off1, co1, tot1, bounds1 = _cum_offset(sp1)
        off2, co2, tot2, bounds2 = _cum_offset(sp2)

        import random
        random.seed(42)
        bg = pairs_all if len(pairs_all) <= 6000 else random.sample(pairs_all, 6000)

        xs_bg, ys_bg = [], []
        for g1, g2, _ in bg:
            info1 = bed[sp1].get(g1)
            info2 = bed[sp2].get(g2)
            if info1 and info2:
                x = off1.get(info1[0], 0) + info1[1] / 1e6
                y = off2.get(info2[0], 0) + info2[1] / 1e6
                xs_bg.append(x)
                ys_bg.append(y)
        ax.scatter(xs_bg, ys_bg, s=0.4, c="#cccccc",
                   linewidths=0, zorder=1, rasterized=True)

        # Chr boundary lines
        for b in bounds1[:-1]:
            ax.axvline(b, color="#e0e0e0", lw=0.5, zorder=2)
        for b in bounds2[:-1]:
            ax.axhline(b, color="#e0e0e0", lw=0.5, zorder=2)

        # Kinesin dots
        both_x, both_y = [], []
        one_x,  one_y  = [], []
        for g1, g2, _ in pairs_all:
            info1 = bed[sp1].get(g1)
            info2 = bed[sp2].get(g2)
            if not info1 or not info2:
                continue
            x = off1.get(info1[0], 0) + info1[1] / 1e6
            y = off2.get(info2[0], 0) + info2[1] / 1e6
            b1 = g1 in kinesin_genes
            b2 = g2 in kinesin_genes
            if b1 and b2:
                both_x.append(x); both_y.append(y)
            elif b1 or b2:
                one_x.append(x);  one_y.append(y)

        ax.scatter(one_x,  one_y,  s=20, c="#f39c12",
                   linewidths=0.5, edgecolors="#c0392b", zorder=3)
        ax.scatter(both_x, both_y, s=40, c="#e74c3c",
                   linewidths=0.7, edgecolors="#922b21", zorder=4,
                   label=f"Both kinesin (n={len(both_x)})")

        # Chr labels on axes (small numbers)
        for ch in co1:
            cx = off1[ch]
            # find end of this chr
            idx_ch = co1.index(ch)
            cx_end = bounds1[idx_ch] if idx_ch < len(bounds1) else tot1
            mid_x  = (cx + cx_end) / 2 - 2.5
            m = re.search(r'(\d+)$', ch)
            num = m.group(1) if m else ch
            ax.text(mid_x, -tot2 * 0.04, num,
                    ha="center", va="top", fontsize=4.5,
                    color=SP_COLOR[sp1], clip_on=False)
        for ch in co2:
            cy = off2[ch]
            idx_ch = co2.index(ch)
            cy_end = bounds2[idx_ch] if idx_ch < len(bounds2) else tot2
            mid_y  = (cy + cy_end) / 2 - 2.5
            m = re.search(r'(\d+)$', ch)
            num = m.group(1) if m else ch
            ax.text(-tot1 * 0.04, mid_y, num,
                    ha="right", va="center", fontsize=4.5,
                    color=SP_COLOR[sp2], clip_on=False)

        ax.set_xlim(-tot1 * 0.02, tot1 * 1.02)
        ax.set_ylim(-tot2 * 0.02, tot2 * 1.02)
        ax.set_xlabel(SP_LABEL[sp1] + " (Mb)", fontsize=7, color=SP_COLOR[sp1])
        ax.set_ylabel(SP_LABEL[sp2] + " (Mb)", fontsize=7, color=SP_COLOR[sp2])
        ax.set_title(f"{sp1} vs {sp2}", fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=5)
        if both_x:
            ax.legend(fontsize=5.5, loc="upper left")

    for idx in range(n_pairs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIGDIR, "mss_3_dotplots.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved →", out)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4  –  MULTI-SPECIES NETWORK
# ══════════════════════════════════════════════════════════════════════════════

def section4_network():
    """
    Network where:
      Nodes  = kinesin-7 genes (coloured and shaped by subgenome)
      Edges  = syntenic anchor connections (one edge per unique gene pair)
    Layout: spring layout approximated by spectral decomposition (no networkx
            required), with subgenome clustering.
    Two panels: (A) full network coloured by subgenome;
                (B) same layout, edge width ∝ score, nodes sized by degree.
    """
    print("\n[Section 4] Multi-species network …")

    # Build adjacency: kinesin gene → set of kinesin partners
    edges = []         # (g1, g2, score)
    edge_set = set()
    for sp1, sp2, fname in ANCHOR_PAIRS:
        for g1, g2, score in anchors.get((sp1, sp2), []):
            if g1 in kinesin_genes and g2 in kinesin_genes:
                key = tuple(sorted([g1, g2]))
                if key not in edge_set:
                    edge_set.add(key)
                    edges.append((g1, g2, score))

    nodes     = sorted(kinesin_genes)
    node_idx  = {g: i for i, g in enumerate(nodes)}
    n         = len(nodes)

    # Degree
    degree = defaultdict(int)
    for g1, g2, _ in edges:
        degree[g1] += 1
        degree[g2] += 1

    # ── Spring layout via iterative force ─────────────────────────────────────
    # Seed positions: arrange subgenomes in sectors of a circle
    subgenome_angle = {
        "Ga":  0,
        "GhA": 60,
        "GbA": 120,
        "Gr":  180,
        "GhD": 240,
        "GbD": 300,
    }
    pos = np.zeros((n, 2))
    rng = np.random.default_rng(42)
    for i, g in enumerate(nodes):
        sp  = sp_of(g) or "Ga"
        ang = np.radians(subgenome_angle.get(sp, 0) + rng.uniform(-25, 25))
        r   = 3.0 + rng.uniform(-0.4, 0.4)
        pos[i] = [r * np.cos(ang), r * np.sin(ang)]

    # Force-directed (Fruchterman–Reingold, simplified)
    area = (n ** 0.5) * 2
    k    = area / n
    temp = 2.0
    for iteration in range(120):
        disp = np.zeros((n, 2))
        # Repulsive forces
        for i in range(n):
            for j in range(i + 1, n):
                delta = pos[i] - pos[j]
                dist  = max(np.linalg.norm(delta), 0.01)
                force = k * k / dist
                unit  = delta / dist
                disp[i] += force * unit
                disp[j] -= force * unit
        # Attractive forces (edges)
        for g1, g2, _ in edges:
            i, j = node_idx[g1], node_idx[g2]
            delta = pos[i] - pos[j]
            dist  = max(np.linalg.norm(delta), 0.01)
            force = dist * dist / k
            unit  = delta / dist
            disp[i] -= force * unit
            disp[j] += force * unit
        # Apply displacement with cooling
        for i in range(n):
            d = np.linalg.norm(disp[i])
            if d > 0:
                pos[i] += (disp[i] / d) * min(d, temp)
        temp = max(temp * 0.92, 0.05)

    # Normalise to [-1, 1]
    pos -= pos.mean(axis=0)
    scale = max(np.abs(pos).max(), 0.1)
    pos /= scale

    # ── Draw ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle("Kinesin-7 Multi-species Synteny Network",
                 fontsize=13, fontweight="bold")

    max_score = max((s for _, _, s in edges), default=1)
    max_deg   = max(degree.values()) if degree else 1

    subgenome_marker = {
        "Ga":  "o", "Gr":  "s",
        "GhA": "^", "GhD": "v",
        "GbA": "D", "GbD": "P",
    }

    for ax_idx, ax in enumerate(axes):
        # Draw edges
        for g1, g2, score in edges:
            i, j   = node_idx[g1], node_idx[g2]
            x_pts  = [pos[i, 0], pos[j, 0]]
            y_pts  = [pos[i, 1], pos[j, 1]]
            sp1_e  = sp_of(g1) or "Ga"
            sp2_e  = sp_of(g2) or "Ga"
            # colour: blend of the two endpoint species colours
            c1 = np.array(matplotlib.colors.to_rgb(SP_COLOR[sp1_e]))
            c2 = np.array(matplotlib.colors.to_rgb(SP_COLOR[sp2_e]))
            ec = tuple((c1 + c2) / 2)
            lw = 0.6 if ax_idx == 0 else 0.4 + 2.5 * score / max_score
            ax.plot(x_pts, y_pts, color=ec, lw=lw, alpha=0.35, zorder=1)

        # Draw nodes
        for i, g in enumerate(nodes):
            sp  = sp_of(g) or "Ga"
            col = SP_COLOR[sp]
            mrk = subgenome_marker.get(sp, "o")
            sz  = 60 if ax_idx == 0 else 30 + 120 * degree[g] / max_deg
            ax.scatter(pos[i, 0], pos[i, 1], s=sz, c=col, marker=mrk,
                       edgecolors="white", linewidths=0.6, zorder=3)

        # Orphan labels (degree 0)
        for i, g in enumerate(nodes):
            if degree[g] == 0:
                sp  = sp_of(g) or "Ga"
                ax.text(pos[i, 0], pos[i, 1] + 0.04, g.split("G")[-1][:6],
                        ha="center", va="bottom", fontsize=4.5,
                        color=SP_COLOR[sp])

        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.axis("off")
        if ax_idx == 0:
            ax.set_title("Nodes coloured by subgenome\n"
                         "(shape = subgenome; orphans labelled)",
                         fontsize=9)
        else:
            ax.set_title("Node size ∝ degree; edge width ∝ synteny score",
                         fontsize=9)

    # Legend
    legend_handles = []
    for sp in SUBGENOMES:
        legend_handles.append(
            plt.scatter([], [], s=60, c=SP_COLOR[sp],
                        marker=subgenome_marker[sp],
                        edgecolors="white", linewidths=0.5,
                        label=SP_LABEL[sp])
        )
    legend_handles.append(
        plt.Line2D([0], [0], lw=1.5, color="#aaaaaa", label="Syntenic edge")
    )
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), ncol=4,
               fontsize=8, frameon=True, framealpha=0.9)

    # Stats annotation
    isolated = sum(1 for g in nodes if degree[g] == 0)
    ax_txt = (
        f"Nodes: {n} kinesin-7 genes   |   "
        f"Edges: {len(edges)} syntenic pairs   |   "
        f"Isolated: {isolated} orphans"
    )
    fig.text(0.5, 0.01, ax_txt, ha="center", fontsize=8, color="#555555")

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = os.path.join(FIGDIR, "mss_4_network.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved →", out)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Kinesin-7 genes loaded: {len(kinesin_genes)}")
    print(f"Subgenomes: {SUBGENOMES}")
    section1_comparative_matrix()
    section2_collinearity()
    section3_dotplots()
    section4_network()
    print("\nAll sections complete.")
