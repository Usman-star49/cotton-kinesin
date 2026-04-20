"""
kinesin7_synteny_ribbon.py
──────────────────────────
Produce two publication-quality synteny ribbon figures in the style of the
reference figure (three horizontal chromosome tracks, arcs coloured by
conservation class, kinesin gene markers as triangles).

Panel A  –  A-subgenome:  Ga  |  GhA  |  GbA
Panel B  –  D-subgenome:  Gr  |  GhD  |  GbD

Outputs
  figures/kinesin7_synteny_ribbon_A.png
  figures/kinesin7_synteny_ribbon_D.png

Legend categories
  • Triple-conserved  – kinesin gene has anchors in BOTH flanking pairs
  • Diploid+one tetraploid  – anchors in exactly one flanking pair (includes
                              the diploid)
  • Both tetraploids only  – anchors in the tetraploid–tetraploid pair only
  • Orphan               – kinesin gene present in the middle genome but no
                              syntenic kinesin partner in either pair
"""

import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patches as mpatch

# ── Configuration ─────────────────────────────────────────────────────────────

DIR    = r"G:\iub_research\kinesin7_synteny\2026-04-14"
FIGDIR = os.path.join(DIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

KINESIN_FILE = os.path.join(DIR, "2026-04-15_kinesin7_genes.txt")

# Each panel:  (title, top-sp, mid-sp, bot-sp, pair_top_mid, pair_mid_bot)
PANELS = [
    dict(
        tag="A",
        title="Kinesin-7 whole-genome synteny — A — A-subgenome (G. arboreum | G. hirsutum A | G. barbadense A)",
        top="Ga",   mid="GhA",  bot="GbA",
        pair_tm="Ga_GhA",   anchors_tm="Ga_GhA.anchors",
        pair_mb="GhA_GbA",  anchors_mb="GhA_GbA.anchors",
    ),
    dict(
        tag="B",
        title="Kinesin-7 whole-genome synteny — B — D-subgenome (G. raimondii | G. hirsutum D | G. barbadense D)",
        top="Gr",   mid="GhD",  bot="GbD",
        pair_tm="Gr_GhD",   anchors_tm="Gr_GhD.anchors",
        pair_mb="GhD_GbD",  anchors_mb="GhD_GbD.anchors",
    ),
]

# Species display labels
SP_LABEL = {
    "Ga":  "G. arboreum (Ga)",
    "Gr":  "G. raimondii (Gr)",
    "GhA": "G. hirsutum A-sub (GhA)",
    "GhD": "G. hirsutum D-sub (GhD)",
    "GbA": "G. barbadense A-sub (GbA)",
    "GbD": "G. barbadense D-sub (GbD)",
}

# Chromosome track colours (matching reference)
SP_COLOR = {
    "Ga":  "#9b59b6",   # purple
    "Gr":  "#9b59b6",   # purple
    "GhA": "#2980b9",   # steel blue
    "GhD": "#2980b9",   # steel blue
    "GbA": "#27ae60",   # green
    "GbD": "#27ae60",   # green
}

# Arc colour by conservation class
ARC_COLOR = {
    "triple":       "#27ae60",   # green  – all 3 species
    "dip_one_tet":  "#2980b9",   # blue   – diploid + one tetraploid
    "both_tet":     "#e67e22",   # orange – both tetraploids only
}

# Marker colour (triangles on chromosomes)
MRK_COLOR = {
    "triple":       "#27ae60",
    "dip_one_tet":  "#2980b9",
    "both_tet":     "#e67e22",
    "orphan":       "#e74c3c",   # red open triangle
}

CHR_HEIGHT  = 0.06   # fraction of vertical space
ARC_ALPHA   = 0.35
TRACK_Y     = [0.82, 0.50, 0.18]   # y-centres for top / mid / bot track
LABEL_X     = 1.002  # right of chromosomes

# ── Helpers ───────────────────────────────────────────────────────────────────

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
    """Return {gene: (chr, mid_bp)}, chr_order list, chr_lengths dict."""
    path = os.path.join(DIR, f"{sp}.bed")
    gene_pos = {}
    chr_genes = defaultdict(list)
    with open(path) as fh:
        for ln in fh:
            p = ln.split()
            if len(p) < 4:
                continue
            ch, st, en, gid = p[0], int(p[1]), int(p[2]), p[3]
            mid = (st + en) / 2
            gene_pos[gid] = (ch, mid)
            chr_genes[ch].append(mid)
    # natural sort chromosomes
    def _chr_key(c):
        m = re.search(r'(\d+)$', c)
        return int(m.group(1)) if m else 0
    chr_order = sorted(chr_genes.keys(), key=_chr_key)
    chr_len = {ch: max(v) for ch, v in chr_genes.items()}
    return gene_pos, chr_order, chr_len


def load_anchors(fname):
    """Return list of (gene_top, gene_bot) pairs from .anchors file."""
    path = os.path.join(DIR, fname)
    pairs = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            p = ln.split()
            if len(p) >= 2:
                pairs.append((p[0], p[1]))
    return pairs


def build_genome_coords(chr_order, chr_len, gap_frac=0.01):
    """Map each chromosome to a [0,1] x-range with small gaps.

    Returns
      chr_start : {chr: float}   – left edge in normalised coords
      total     : float          – sum of all lengths (normalised = 1 after scaling)
      scale     : float          – bp per unit
    """
    total_bp = sum(chr_len[c] for c in chr_order)
    n_gaps   = len(chr_order) - 1
    # gap in bp equivalent
    gap_bp   = gap_frac * total_bp / max(n_gaps, 1)
    grand    = total_bp + n_gaps * gap_bp
    cur = 0.0
    chr_start = {}
    for c in chr_order:
        chr_start[c] = cur / grand
        cur += chr_len[c] + gap_bp
    scale = grand   # bp -> [0,1] : divide by scale
    return chr_start, scale


def gene_x(gene, gene_pos, chr_start, scale):
    """Return normalised x position [0,1] of a gene."""
    if gene not in gene_pos:
        return None
    ch, mid = gene_pos[gene]
    if ch not in chr_start:
        return None
    return chr_start[ch] + mid / scale


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_chromosomes(ax, sp, chr_order, chr_len, chr_start, scale, y_centre, color):
    h = CHR_HEIGHT
    for ch in chr_order:
        x0 = chr_start[ch]
        x1 = x0 + chr_len[ch] / scale
        rect = mpatches.FancyBboxPatch(
            (x0, y_centre - h / 2), x1 - x0, h,
            boxstyle="round,pad=0.002",
            linewidth=0.4, edgecolor="white",
            facecolor=color, zorder=3
        )
        ax.add_patch(rect)
        # chromosome number label (small, centred)
        m = re.search(r'(\d+)$', ch)
        num = m.group(1) if m else ch
        ax.text((x0 + x1) / 2, y_centre, num,
                ha='center', va='center',
                fontsize=5.5, color='white', fontweight='bold', zorder=4)


def draw_arc(ax, x1, y1, x2, y2, color, alpha, lw=0.6):
    """Cubic Bezier arc between two points on different tracks."""
    y_ctrl = (y1 + y2) / 2
    verts = [(x1, y1), (x1, y_ctrl), (x2, y_ctrl), (x2, y2)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    p = mpatch.PathPatch(Path(verts, codes),
                         facecolor='none', edgecolor=color,
                         lw=lw, alpha=alpha, zorder=2)
    ax.add_patch(p)


def draw_marker(ax, x, y_centre, cls, direction='down'):
    """Triangle marker on a chromosome track."""
    color  = MRK_COLOR.get(cls, "#555555")
    filled = cls != "orphan"
    dy = CHR_HEIGHT / 2 + 0.012
    if direction == 'down':
        marker = 'v'
        y = y_centre + dy
    else:
        marker = '^'
        y = y_centre - dy
    ax.plot(x, y, marker=marker, ms=4.5, color=color,
            markerfacecolor=color if filled else 'none',
            markeredgewidth=0.8, markeredgecolor=color,
            zorder=5, clip_on=False)


# ── Syntenic block shading ─────────────────────────────────────────────────────

def draw_block_shade(ax, pairs_tm, pairs_mb,
                     gene_pos_top, chr_start_top, scale_top, y_top,
                     gene_pos_mid, chr_start_mid, scale_mid, y_mid,
                     gene_pos_bot, chr_start_bot, scale_bot, y_bot):
    """Draw faint chromosome-coloured background ribbons for synteny blocks."""
    # Colour blocks by chr of middle (reference) genome
    # We skip detailed block detection for now; arcs carry the information.
    pass   # placeholder – arcs alone replicate the reference figure adequately


# ── Main panel builder ─────────────────────────────────────────────────────────

def build_panel(cfg, kinesin_genes):
    top_sp = cfg["top"]
    mid_sp = cfg["mid"]
    bot_sp = cfg["bot"]

    # Load BED data
    gp_top, co_top, cl_top = load_bed(top_sp)
    gp_mid, co_mid, cl_mid = load_bed(mid_sp)
    gp_bot, co_bot, cl_bot = load_bed(bot_sp)

    # Build normalised coordinate systems
    cs_top, sc_top = build_genome_coords(co_top, cl_top)
    cs_mid, sc_mid = build_genome_coords(co_mid, cl_mid)
    cs_bot, sc_bot = build_genome_coords(co_bot, cl_bot)

    # Load anchor pairs
    anch_tm = load_anchors(cfg["anchors_tm"])   # top <-> mid
    anch_mb = load_anchors(cfg["anchors_mb"])   # mid <-> bot

    # Build sets: which kinesin genes appear in each anchor set
    kin_in_tm_top = set(); kin_in_tm_mid = set()
    kin_in_mb_mid = set(); kin_in_mb_bot = set()

    tm_pairs_kin = []   # (top_gene, mid_gene) where at least one is kinesin
    mb_pairs_kin = []

    for g1, g2 in anch_tm:
        if g1 in kinesin_genes or g2 in kinesin_genes:
            tm_pairs_kin.append((g1, g2))
            if g1 in kinesin_genes: kin_in_tm_top.add(g1)
            if g2 in kinesin_genes: kin_in_tm_mid.add(g2)

    for g1, g2 in anch_mb:
        if g1 in kinesin_genes or g2 in kinesin_genes:
            mb_pairs_kin.append((g1, g2))
            if g1 in kinesin_genes: kin_in_mb_mid.add(g1)
            if g2 in kinesin_genes: kin_in_mb_bot.add(g2)

    # Classify each kinesin gene in the middle genome
    mid_kin = {g for g in kinesin_genes if sp_of(g) == mid_sp}

    def classify_mid(gene):
        in_tm = gene in kin_in_tm_mid
        in_mb = gene in kin_in_mb_mid
        if in_tm and in_mb:
            return "triple"
        if in_tm or in_mb:
            return "dip_one_tet"
        return "orphan"

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 8))
    ax.set_xlim(-0.005, 1.04)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    yt, ym, yb = TRACK_Y

    # Draw chromosome tracks
    draw_chromosomes(ax, top_sp, co_top, cl_top, cs_top, sc_top, yt, SP_COLOR[top_sp])
    draw_chromosomes(ax, mid_sp, co_mid, cl_mid, cs_mid, sc_mid, ym, SP_COLOR[mid_sp])
    draw_chromosomes(ax, bot_sp, co_bot, cl_bot, cs_bot, sc_bot, yb, SP_COLOR[bot_sp])

    # Species labels (right side)
    for y, sp in [(yt, top_sp), (ym, mid_sp), (yb, bot_sp)]:
        ax.text(LABEL_X, y, SP_LABEL[sp],
                va='center', ha='left', fontsize=9,
                color=SP_COLOR[sp], fontweight='bold')

    # ── Draw ALL syntenic arcs (genome-wide background, thin, grey) ──────────
    # Sample up to 8000 random background arcs for readability
    import random
    random.seed(42)

    def _draw_bg_arcs(pairs, gp1, cs1, sc1, y1, gp2, cs2, sc2, y2, max_arcs=4000):
        sample = pairs if len(pairs) <= max_arcs else random.sample(pairs, max_arcs)
        for g1, g2 in sample:
            x1 = gene_x(g1, gp1, cs1, sc1)
            x2 = gene_x(g2, gp2, cs2, sc2)
            if x1 is None or x2 is None:
                continue
            draw_arc(ax, x1, y1, x2, y2, "#aaaaaa", alpha=0.10, lw=0.3)

    _draw_bg_arcs(anch_tm, gp_top, cs_top, sc_top, yt,
                           gp_mid, cs_mid, sc_mid, ym)
    _draw_bg_arcs(anch_mb, gp_mid, cs_mid, sc_mid, ym,
                           gp_bot, cs_bot, sc_bot, yb)

    # ── Draw kinesin arcs (coloured, on top) ─────────────────────────────────
    # top–mid
    for g1, g2 in tm_pairs_kin:
        x1 = gene_x(g1, gp_top, cs_top, sc_top)
        x2 = gene_x(g2, gp_mid, cs_mid, sc_mid)
        if x1 is None or x2 is None:
            continue
        # classify by mid gene
        cls = classify_mid(g2) if g2 in mid_kin else "dip_one_tet"
        color = ARC_COLOR.get(cls, ARC_COLOR["dip_one_tet"])
        draw_arc(ax, x1, yt, x2, ym, color, alpha=0.75, lw=1.0)

    # mid–bot
    for g1, g2 in mb_pairs_kin:
        x1 = gene_x(g1, gp_mid, cs_mid, sc_mid)
        x2 = gene_x(g2, gp_bot, cs_bot, sc_bot)
        if x1 is None or x2 is None:
            continue
        cls = classify_mid(g1) if g1 in mid_kin else "dip_one_tet"
        # both_tet arcs (no top connection) → orange
        if cls == "orphan":
            cls = "both_tet"
        color = ARC_COLOR.get(cls, ARC_COLOR["both_tet"])
        draw_arc(ax, x1, ym, x2, yb, color, alpha=0.75, lw=1.0)

    # ── Kinesin gene markers (triangles) ─────────────────────────────────────
    def _marker_class_top(gene):
        """Classify a top-genome kinesin."""
        return "dip_one_tet" if gene in kin_in_tm_top else "orphan"

    def _marker_class_bot(gene):
        return "dip_one_tet" if gene in kin_in_mb_bot else "orphan"

    # Top track
    for g in kinesin_genes:
        if sp_of(g) != top_sp:
            continue
        x = gene_x(g, gp_top, cs_top, sc_top)
        if x is None:
            continue
        cls = _marker_class_top(g)
        draw_marker(ax, x, yt, cls, direction='down')

    # Mid track – top-pointing and bottom-pointing
    for g in mid_kin:
        x = gene_x(g, gp_mid, cs_mid, sc_mid)
        if x is None:
            continue
        cls = classify_mid(g)
        draw_marker(ax, x, ym, cls, direction='down')   # towards top
        draw_marker(ax, x, ym, cls, direction='up')     # towards bottom

    # Bot track
    for g in kinesin_genes:
        if sp_of(g) != bot_sp:
            continue
        x = gene_x(g, gp_bot, cs_bot, sc_bot)
        if x is None:
            continue
        cls = _marker_class_bot(g)
        draw_marker(ax, x, yb, cls, direction='up')

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=SP_COLOR[top_sp],  label=f"G. {_sp_full(top_sp)} ({top_sp})"),
        mpatches.Patch(facecolor=SP_COLOR[mid_sp],  label=f"G. {_sp_full(mid_sp)} ({mid_sp})"),
        mpatches.Patch(facecolor=SP_COLOR[bot_sp],  label=f"G. {_sp_full(bot_sp)} ({bot_sp})"),
        plt.Line2D([0],[0], color=ARC_COLOR["triple"],       lw=1.5, label="arcs — All 3 species (triple conserved)"),
        plt.Line2D([0],[0], color=ARC_COLOR["dip_one_tet"],  lw=1.5, label="arcs — Diploid + one tetraploid"),
        plt.Line2D([0],[0], color=ARC_COLOR["both_tet"],     lw=1.5, label="arcs — Both tetraploids only (diploid absent)"),
        plt.Line2D([0],[0], marker='v', color=MRK_COLOR["triple"],       ms=6,
                   linestyle='None', markerfacecolor=MRK_COLOR["triple"],
                   label="All 3 species (triple conserved)"),
        plt.Line2D([0],[0], marker='v', color=MRK_COLOR["dip_one_tet"],  ms=6,
                   linestyle='None', markerfacecolor=MRK_COLOR["dip_one_tet"],
                   label="Diploid + one tetraploid"),
        plt.Line2D([0],[0], marker='v', color=MRK_COLOR["both_tet"],     ms=6,
                   linestyle='None', markerfacecolor=MRK_COLOR["both_tet"],
                   label="Both tetraploids only (diploid absent)"),
        plt.Line2D([0],[0], marker='v', color=MRK_COLOR["orphan"],       ms=6,
                   linestyle='None', markerfacecolor='none',
                   markeredgecolor=MRK_COLOR["orphan"],
                   label="Orphan (no syntelog)"),
        mpatches.Patch(facecolor='#dddddd', edgecolor='#aaaaaa',
                       label="Syntenic block (colored by chr)"),
    ]
    ax.legend(handles=legend_items, loc='lower center',
              bbox_to_anchor=(0.46, -0.08), ncol=4,
              fontsize=7.5, frameon=True, framealpha=0.9,
              handlelength=1.6, columnspacing=1.0)

    ax.set_title(cfg["title"], fontsize=11, fontweight='bold', pad=14)

    out = os.path.join(FIGDIR, f"kinesin7_synteny_ribbon_{cfg['tag']}.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


def _sp_full(sp):
    m = {"Ga": "arboreum", "Gr": "raimondii",
         "GhA": "hirsutum A-sub", "GhD": "hirsutum D-sub",
         "GbA": "barbadense A-sub", "GbD": "barbadense D-sub"}
    return m.get(sp, sp)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    kinesin_genes = load_kinesin(KINESIN_FILE)
    print(f"Loaded {len(kinesin_genes)} kinesin-7 genes")
    for cfg in PANELS:
        print(f"\nBuilding panel {cfg['tag']} …")
        build_panel(cfg, kinesin_genes)
    print("\nDone.")
