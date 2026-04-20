"""
kinesin7_synteny_analysis.py
────────────────────────────
Main analysis engine for Kinesin-7 gene family in four cotton species.

Sections
  1. Configuration
  2. Data-loading helpers
  3. Intra-species synteny detection
  4. Duplication classification  (Tandem / Proximal / Segmental / Homeolog / Ortholog)
  5. Ka/Ks estimation            (Nei-Gojobori 1986, protein-guided codon alignment)
  6. Visualization
  7. Main pipeline

Inputs  (DIR = G:/iub_research/kinesin7_synteny/2026-04-14)
  2026-04-15_kinesin7_genes.txt
  *.bed                   -- Ga  GhA  GhD  GbA  GbD  Gr
  *_*.anchors             -- 6 inter-species + 2 homeolog pairs
  New folder/*.cds*       -- CDS FASTA for Ga, Gh, Gb, Gr

Outputs  (written to DIR  /  figures subfolder)
  kinesin7_duplication_unified.tsv
  kinesin7_kaks.tsv
  figures/kinesin7_dup_summary.png
  figures/kinesin7_kaks_distribution.png
  figures/kinesin7_kaks_vs_category.png
"""

import math
import os
import re
import warnings
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# ==============================================================================
# SECTION 1 -- CONFIGURATION
# ==============================================================================

DIR = r"G:\iub_research\kinesin7_synteny\2026-04-14"
CDS_DIR = os.path.join(DIR, "New folder")
FIGDIR = os.path.join(DIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

KINESIN_FILE = os.path.join(DIR, "2026-04-15_kinesin7_genes.txt")

# (query_subgenome, subject_subgenome) -> relationship type
ANCHORS_CONFIG = {
    ("Ga",  "GhA"): "inter",
    ("Ga",  "GbA"): "inter",
    ("GhA", "GbA"): "inter",
    ("Gr",  "GhD"): "inter",
    ("Gr",  "GbD"): "inter",
    ("GhD", "GbD"): "inter",
    ("GhA", "GhD"): "homeo",   # within Gh tetraploid
    ("GbA", "GbD"): "homeo",   # within Gb tetraploid
}

BED_LABELS = ["Ga", "GhA", "GhD", "GbA", "GbD", "Gr"]

# One CDS file per species (tetraploid files contain both subgenomes)
CDS_FILES = {
    "Ga": os.path.join(CDS_DIR, "Garboreum_A2_CRI.cds_new_fa.cds"),
    "Gh": os.path.join(CDS_DIR, "Ghirsutum_578_v3.1_new.cds.fa"),
    "Gb": os.path.join(CDS_DIR, "Gbarbadense_526_v1.1.cds.fa"),
    "Gr": os.path.join(CDS_DIR, "Graimondii_221__new.cds.fa"),
}

TANDEM_THRESH   = 20    # gene-rank distance for tandem
PROXIMAL_THRESH = 200   # gene-rank distance for proximal

CATEGORIES = ["Ortholog", "Homeolog", "Segmental", "Tandem", "Proximal", "Dispersed"]
CAT_COLORS = {
    "Ortholog":  "#3498db",
    "Homeolog":  "#e74c3c",
    "Segmental": "#2ecc71",
    "Tandem":    "#f39c12",
    "Proximal":  "#9b59b6",
    "Dispersed": "#95a5a6",
}

# ==============================================================================
# SECTION 2 -- DATA-LOADING HELPERS
# ==============================================================================

def load_kinesin_genes(path):
    with open(path) as fh:
        return {ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")}


def species_of(gene):
    """Two-letter species code: Gh / Gb / Gr / Ga."""
    if gene.startswith("GohirA") or gene.startswith("GohirD"):
        return "Gh"
    if gene.startswith("GobarA") or gene.startswith("GobarD"):
        return "Gb"
    if gene.startswith("Gorai"):
        return "Gr"
    if gene.startswith("Ga"):
        return "Ga"
    return "?"


def subgenome_of(gene):
    """Subgenome label: GhA / GhD / GbA / GbD / Ga / Gr."""
    if gene.startswith("GohirA"):
        return "GhA"
    if gene.startswith("GohirD"):
        return "GhD"
    if gene.startswith("GobarA"):
        return "GbA"
    if gene.startswith("GobarD"):
        return "GbD"
    if gene.startswith("Gorai"):
        return "Gr"
    if gene.startswith("Ga"):
        return "Ga"
    return "?"


def load_bed_files(labels, base_dir):
    """Build gene_chrom and gene_rank (local rank on each chromosome)."""
    gene_chrom = {}
    chr_gene_list = defaultdict(list)

    for label in labels:
        path = os.path.join(base_dir, f"{label}.bed")
        if not os.path.exists(path):
            print(f"  [WARN] BED not found: {path}")
            continue
        with open(path) as fh:
            for line in fh:
                p = line.split()
                if len(p) < 4:
                    continue
                chrom, start, gene = p[0], int(p[1]), p[3]
                gene_chrom[gene] = chrom
                chr_gene_list[chrom].append((start, gene))

    gene_rank = {}
    for chrom, items in chr_gene_list.items():
        items.sort()
        for rank, (_, gene) in enumerate(items):
            gene_rank[gene] = rank

    return gene_chrom, gene_rank


def load_all_anchors(anchors_config, base_dir):
    """
    Load all anchors files.

    Returns
    -------
    pair_evidence : dict  frozenset({gA, gB}) -> set of "type:filename" tags
    synteny_pairs : set   {(gA, gB), (gB, gA)} for O(1) membership testing
    """
    pair_evidence = defaultdict(set)
    synteny_pairs = set()

    for (q, s), ptype in anchors_config.items():
        path = os.path.join(base_dir, f"{q}_{s}.anchors")
        if not os.path.exists(path):
            print(f"  [WARN] anchors not found: {path}")
            continue
        count = 0
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                gA, gB = parts[0], parts[1]
                key = frozenset({gA, gB})
                pair_evidence[key].add(f"{ptype}:{q}_{s}.anchors")
                synteny_pairs.add((gA, gB))
                synteny_pairs.add((gB, gA))
                count += 1
        print(f"  {q}_{s}.anchors [{ptype}]  {count:>8,} pairs")

    print(f"  -> {len(pair_evidence):,} unique anchored pairs total")
    return dict(pair_evidence), synteny_pairs


# ==============================================================================
# SECTION 3 -- INTRA-SPECIES SYNTENY DETECTION
# ==============================================================================

def detect_intra_species_synteny(kinesin_set, pair_evidence, gene_chrom,
                                  gene_rank, anchors_config):
    """
    Identify kinesin genes with intra-species syntenic partners.

    Strategy
    --------
    - Tetraploids (Gh, Gb): homeolog anchors define intra-species synteny
      between the two subgenomes.
    - All species: tandem/proximal kinesin neighbours on the same chromosome
      are included regardless of anchor coverage.

    Returns
    -------
    intra : dict  gene -> list of {"partner", "type", "evidence"} records
    """
    intra = defaultdict(list)

    # (a) Homeolog pairs within each tetraploid
    for key, evs in pair_evidence.items():
        gene_list = list(key)
        if len(gene_list) != 2:
            continue
        gA, gB = gene_list
        if gA not in kinesin_set or gB not in kinesin_set:
            continue
        if not any("homeo" in e for e in evs):
            continue
        spA, spB = species_of(gA), species_of(gB)
        if spA == spB and spA in ("Gh", "Gb"):
            tag = ";".join(sorted(evs))
            intra[gA].append({"partner": gB, "type": "Homeolog", "evidence": tag})
            intra[gB].append({"partner": gA, "type": "Homeolog", "evidence": tag})

    # (b) Tandem / proximal neighbours on the same chromosome
    chr_kin = defaultdict(list)
    for gene in kinesin_set:
        chrom = gene_chrom.get(gene)
        if chrom:
            chr_kin[chrom].append((gene_rank.get(gene, 0), gene))

    for chrom, items in chr_kin.items():
        items.sort()
        for i, (ri, gi) in enumerate(items):
            for rj, gj in items[i + 1:]:
                dist = rj - ri
                if dist > PROXIMAL_THRESH:
                    break
                if any(e["partner"] == gj for e in intra[gi]):
                    continue
                dup_type = "Tandem" if dist <= TANDEM_THRESH else "Proximal"
                tag = f"BED:{chrom}:dist={dist}"
                intra[gi].append({"partner": gj, "type": dup_type, "evidence": tag})
                intra[gj].append({"partner": gi, "type": dup_type, "evidence": tag})

    return dict(intra)


# ==============================================================================
# SECTION 4 -- DUPLICATION CLASSIFICATION
# ==============================================================================

def classify_pair(gA, gB, gene_chrom, gene_rank, synteny_pairs):
    """
    Assign a duplication category.
    Priority: Homeolog > Ortholog > Segmental > Tandem > Proximal > Dispersed
    """
    spA, spB = species_of(gA), species_of(gB)
    sgA, sgB = subgenome_of(gA), subgenome_of(gB)
    in_synteny = (gA, gB) in synteny_pairs

    # Homeolog: same tetraploid, different subgenomes, anchored
    if spA == spB and spA in ("Gh", "Gb") and sgA != sgB and in_synteny:
        return "Homeolog"

    # Ortholog: different species, anchored
    if spA != spB and in_synteny:
        return "Ortholog"

    # Different species, no anchor
    if spA != spB:
        return "Dispersed"

    # Same species below this line
    cA = gene_chrom.get(gA)
    cB = gene_chrom.get(gB)

    # Segmental (WGD paralog): same species, different chromosomes, anchored
    if in_synteny and cA != cB:
        return "Segmental"

    if cA and cB and cA == cB:
        dist = abs(gene_rank.get(gA, -1) - gene_rank.get(gB, -1))
        if dist <= TANDEM_THRESH:
            return "Tandem"
        if dist <= PROXIMAL_THRESH:
            return "Proximal"

    return "Dispersed"


def build_duplication_table(kinesin_set, pair_evidence, gene_chrom,
                             gene_rank, synteny_pairs):
    """
    Build a unified duplication classification table covering:
      - All kinesin x kinesin pairs present in any anchors file.
      - BED-derived tandem/proximal pairs not in any anchor.
    """
    results = []
    seen = set()

    # (a) Anchor-supported pairs
    for key, evs in pair_evidence.items():
        gene_list = list(key)
        if len(gene_list) != 2:
            continue
        gA, gB = sorted(gene_list)
        if gA not in kinesin_set or gB not in kinesin_set:
            continue
        pair_key = (gA, gB)
        if pair_key in seen:
            continue
        seen.add(pair_key)
        cat = classify_pair(gA, gB, gene_chrom, gene_rank, synteny_pairs)
        results.append({
            "GeneA":    gA,
            "GeneB":    gB,
            "SpeciesA": species_of(gA),
            "SubA":     subgenome_of(gA),
            "SpeciesB": species_of(gB),
            "SubB":     subgenome_of(gB),
            "ChrA":     gene_chrom.get(gA, "?"),
            "ChrB":     gene_chrom.get(gB, "?"),
            "Category": cat,
            "Evidence": ";".join(sorted(evs)),
        })

    # (b) BED-only tandem / proximal pairs
    chr_kin = defaultdict(list)
    for gene in kinesin_set:
        chrom = gene_chrom.get(gene)
        if chrom:
            chr_kin[chrom].append((gene_rank.get(gene, 0), gene))

    for chrom, items in chr_kin.items():
        items.sort()
        for i, (ri, gi) in enumerate(items):
            for rj, gj in items[i + 1:]:
                dist = rj - ri
                if dist > PROXIMAL_THRESH:
                    break
                pair_key = tuple(sorted([gi, gj]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                cat = "Tandem" if dist <= TANDEM_THRESH else "Proximal"
                gA, gB = pair_key
                results.append({
                    "GeneA":    gA,
                    "GeneB":    gB,
                    "SpeciesA": species_of(gA),
                    "SubA":     subgenome_of(gA),
                    "SpeciesB": species_of(gB),
                    "SubB":     subgenome_of(gB),
                    "ChrA":     chrom,
                    "ChrB":     chrom,
                    "Category": cat,
                    "Evidence": f"BED:{chrom}:dist={dist}",
                })

    return results


# ==============================================================================
# SECTION 5 -- Ka/Ks ESTIMATION  (Nei-Gojobori 1986)
# ==============================================================================

_CODON_AA = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
_BASES = "ACGT"


def _syn_sites(codon):
    """Count synonymous and nonsynonymous sites (Nei-Gojobori 1986)."""
    codon = codon.upper()
    if codon not in _CODON_AA or _CODON_AA[codon] == "*":
        return 0.0, 0.0
    ref_aa = _CODON_AA[codon]
    s = 0.0
    for pos in range(3):
        for base in _BASES:
            if base == codon[pos]:
                continue
            mut = codon[:pos] + base + codon[pos + 1:]
            if _CODON_AA.get(mut) == ref_aa:
                s += 1 / 3
    return s, 3.0 - s


def _jc_correction(p):
    """Jukes-Cantor distance from proportion of differences p."""
    if p >= 0.75:
        return float("inf")
    x = 1.0 - (4.0 / 3.0) * p
    if x <= 0:
        return float("inf")
    return -0.75 * math.log(x)


def nei_gojobori(seq1, seq2):
    """
    Ka and Ks for two codon-aligned CDS strings (equal length, no gaps).
    Nei-Gojobori (1986) counting method with Jukes-Cantor correction.
    Returns (Ka, Ks, omega) or (None, None, None) on failure.
    """
    if len(seq1) != len(seq2) or len(seq1) % 3 != 0 or len(seq1) == 0:
        return None, None, None

    S_total = N_total = sd = nd = 0.0

    for i in range(0, len(seq1), 3):
        c1 = seq1[i: i + 3].upper()
        c2 = seq2[i: i + 3].upper()
        if any(b not in _BASES for b in c1 + c2):
            continue
        if _CODON_AA.get(c1) == "*" or _CODON_AA.get(c2) == "*":
            continue

        s1, n1 = _syn_sites(c1)
        s2, n2 = _syn_sites(c2)
        S_total += (s1 + s2) / 2
        N_total += (n1 + n2) / 2

        diffs = sum(a != b for a, b in zip(c1, c2))
        if diffs == 0:
            continue
        elif diffs == 1:
            if _CODON_AA.get(c1) == _CODON_AA.get(c2):
                sd += 1
            else:
                nd += 1
        else:
            # Multiple differences: proportion-weighted split
            frac_s = (s1 + s2) / 6.0
            sd += diffs * frac_s
            nd += diffs * (1.0 - frac_s)

    if S_total < 1e-6 or N_total < 1e-6:
        return None, None, None

    ks = _jc_correction(sd / S_total)
    ka = _jc_correction(nd / N_total)

    if ks == float("inf") or ka == float("inf") or ks < 0 or ka < 0:
        return None, None, None

    omega = round(ka / ks, 4) if ks > 1e-9 else None
    return round(ka, 4), round(ks, 4), omega


def _align_cds_by_protein(cds1, cds2):
    """
    Protein-guided codon alignment using Biopython PairwiseAligner.
    Translates both CDS, globally aligns the proteins, maps columns back
    to codons, then strips gapped positions.
    Falls back to length-truncation if Biopython is unavailable.
    Returns (aligned_cds1, aligned_cds2).
    """
    try:
        from Bio.Align import PairwiseAligner
        from Bio.Seq import Seq

        prot1 = str(Seq(cds1).translate(to_stop=True))
        prot2 = str(Seq(cds2).translate(to_stop=True))
        if not prot1 or not prot2:
            raise ValueError("empty protein")

        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2
        aligner.mismatch_score = -1
        aligner.open_gap_score = -3
        aligner.extend_gap_score = -0.5

        alignment = next(iter(aligner.align(prot1, prot2)))
        aln1 = alignment[0]
        aln2 = alignment[1]

        codons1 = [cds1[i: i + 3] for i in range(0, len(prot1) * 3, 3)]
        codons2 = [cds2[i: i + 3] for i in range(0, len(prot2) * 3, 3)]

        out1, out2 = [], []
        idx1 = idx2 = 0
        for aa1, aa2 in zip(aln1, aln2):
            c1 = codons1[idx1] if aa1 != "-" and idx1 < len(codons1) else None
            c2 = codons2[idx2] if aa2 != "-" and idx2 < len(codons2) else None
            if aa1 != "-":
                idx1 += 1
            if aa2 != "-":
                idx2 += 1
            if c1 is not None and c2 is not None:
                out1.append(c1)
                out2.append(c2)

        return "".join(out1), "".join(out2)

    except Exception:
        min_len = (min(len(cds1), len(cds2)) // 3) * 3
        return cds1[:min_len], cds2[:min_len]


def load_cds_index(cds_files, kinesin_set):
    """
    Parse CDS FASTA files and index sequences for kinesin genes.

    Header normalisation
    --------------------
    Ga:       >evm.model.Ga06G0266      ->  strip 'evm.model.' prefix
    Gh/Gb/Gr: >Gohir.A11G307601.1      ->  remove dot after species prefix,
                                            strip trailing isoform '.N' suffix
              >Gohir.1Z049314.1         ->  becomes Gohir1Z049314 which is
                                            not in kinesin_set -> skipped
    """
    try:
        from Bio import SeqIO
    except ImportError:
        print("  [WARN] Biopython not installed -- CDS loading skipped")
        return {}

    cds_index = {}
    for sp, path in cds_files.items():
        if not os.path.exists(path):
            print(f"  [WARN] CDS file not found: {path}")
            continue
        print(f"  Indexing CDS ({sp}): {os.path.basename(path)} ...")
        count = 0
        try:
            for record in SeqIO.parse(path, "fasta"):
                gene_id = record.id
                # Ga: strip 'evm.model.' prefix -> Ga01G1298
                gene_id = re.sub(r"^evm\.model\.", "", gene_id)
                # Gh/Gb/Gr: remove the dot between species prefix and
                # chromosome/gene -> Gohir.A11G307601 -> GohirA11G307601
                gene_id = re.sub(r"^([A-Za-z]+)\.", r"\1", gene_id)
                # Strip trailing isoform suffix (.1, .2, ...)
                gene_id = re.sub(r"\.\d+$", "", gene_id)
                if gene_id not in kinesin_set:
                    continue
                seq = str(record.seq).upper().replace(" ", "").replace("\n", "")
                seq = seq[: (len(seq) // 3) * 3]
                # Keep the longest isoform if multiple entries share an ID
                if gene_id not in cds_index or len(seq) > len(cds_index[gene_id]):
                    cds_index[gene_id] = seq
                    count += 1
        except Exception as exc:
            print(f"    [ERROR] {exc}")
        print(f"    -> {count} kinesin CDS loaded ({sp})")

    return cds_index


def compute_kaks(dup_table, cds_index):
    """
    Compute Ka, Ks, and Ka/Ks for every pair in dup_table that has CDS data.
    Returns a list of dicts (dup_table rows with Ka, Ks, Omega columns added).
    """
    kaks_rows = []
    n_total = n_ok = n_missing = n_fail = 0

    for row in dup_table:
        gA, gB = row["GeneA"], row["GeneB"]
        n_total += 1
        seq1 = cds_index.get(gA)
        seq2 = cds_index.get(gB)
        if seq1 is None or seq2 is None:
            n_missing += 1
            continue
        s1, s2 = _align_cds_by_protein(seq1, seq2)
        ka, ks, omega = nei_gojobori(s1, s2)
        if ka is not None:
            n_ok += 1
            kaks_rows.append({**row, "Ka": ka, "Ks": ks, "Omega": omega})
        else:
            n_fail += 1

    print(
        f"  Ka/Ks: {n_ok} computed  |  {n_missing} missing CDS  "
        f"|  {n_fail} failed  |  {n_total} total pairs"
    )
    return kaks_rows


# ==============================================================================
# SECTION 6 -- VISUALIZATION
# ==============================================================================

def plot_duplication_summary(dup_table, figdir):
    """Horizontal stacked bar per species pair + overall pie chart."""
    sp_pair_counts = defaultdict(Counter)
    for row in dup_table:
        key = "-".join(sorted({row["SpeciesA"], row["SpeciesB"]}))
        sp_pair_counts[key][row["Category"]] += 1

    sp_keys = sorted(sp_pair_counts)
    n = len(sp_keys)

    fig, axes = plt.subplots(
        1, 2, figsize=(15, max(5, n * 0.45 + 2)),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    ax = axes[0]
    bottoms = np.zeros(n)
    for cat in CATEGORIES:
        vals = np.array([sp_pair_counts[k].get(cat, 0) for k in sp_keys], dtype=float)
        ax.barh(sp_keys, vals, left=bottoms,
                color=CAT_COLORS.get(cat, "#888888"),
                label=cat, height=0.65, edgecolor="white", linewidth=0.4)
        bottoms += vals
    ax.set_xlabel("Kinesin-7 gene pairs", fontsize=11)
    ax.set_title("Duplication category by species pair", fontsize=12, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(axis="x", linewidth=0.3, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    overall = Counter(row["Category"] for row in dup_table)
    wedge_labels = [c for c in CATEGORIES if overall.get(c, 0) > 0]
    wedge_vals   = [overall[c] for c in wedge_labels]
    wedge_colors = [CAT_COLORS.get(c, "#888888") for c in wedge_labels]
    axes[1].pie(wedge_vals, labels=wedge_labels, colors=wedge_colors,
                autopct="%1.0f%%", startangle=140, textprops={"fontsize": 8})
    axes[1].set_title("Overall", fontsize=11)

    plt.tight_layout()
    out = os.path.join(figdir, "kinesin7_dup_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_kaks_distribution(kaks_rows, figdir):
    """Histograms of Ka, Ks, and Ka/Ks (omega)."""
    kas    = [r["Ka"]    for r in kaks_rows if r["Ka"]    is not None]
    kss    = [r["Ks"]    for r in kaks_rows if r["Ks"]    is not None]
    omegas = [r["Omega"] for r in kaks_rows if r["Omega"] is not None]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, label, color in [
        (axes[0], kas,    "Ka (nonsynonymous)", "#e74c3c"),
        (axes[1], kss,    "Ks (synonymous)",    "#3498db"),
        (axes[2], omegas, "Ka/Ks (omega)",      "#2ecc71"),
    ]:
        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(label, fontsize=10)
            continue
        ax.hist(data, bins=30, color=color, edgecolor="white",
                linewidth=0.5, alpha=0.85)
        ax.axvline(float(np.median(data)), color="k", linestyle="--",
                   linewidth=1, label=f"Median={np.median(data):.3f}")
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Pairs", fontsize=10)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Kinesin-7 substitution rates", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(figdir, "kinesin7_kaks_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_kaks_by_category(kaks_rows, figdir):
    """Violin + jitter of Ka/Ks (omega) grouped by duplication category."""
    cat_omegas = defaultdict(list)
    for row in kaks_rows:
        if row["Omega"] is not None:
            cat_omegas[row["Category"]].append(row["Omega"])

    cats = [c for c in CATEGORIES if cat_omegas.get(c)]
    if not cats:
        print("  [SKIP] kinesin7_kaks_vs_category.png -- no omega values")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(
        [cat_omegas[c] for c in cats],
        positions=list(range(len(cats))),
        showmedians=True,
        showextrema=True,
    )
    for body, cat in zip(parts["bodies"], cats):
        body.set_facecolor(CAT_COLORS.get(cat, "#888888"))
        body.set_alpha(0.7)

    rng = np.random.default_rng(seed=42)
    for i, cat in enumerate(cats):
        vals = cat_omegas[cat]
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=CAT_COLORS.get(cat, "#888888"),
                   alpha=0.5, s=12, zorder=3)

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1,
               label="omega = 1 (neutral)")
    ax.set_xticks(list(range(len(cats))))
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylabel("Ka/Ks (omega)", fontsize=11)
    ax.set_title("Ka/Ks by duplication category", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(figdir, "kinesin7_kaks_vs_category.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ==============================================================================
# SECTION 7 -- MAIN PIPELINE
# ==============================================================================

def main():
    print("=" * 60)
    print(" Kinesin-7 Synteny Analysis -- Main Engine")
    print("=" * 60)

    # -- 1. Core data ----------------------------------------------------------
    print("\n[1] Loading kinesin gene set ...")
    kinesin_set = load_kinesin_genes(KINESIN_FILE)
    print(f"    {len(kinesin_set)} kinesin-7 genes")

    print("\n[2] Loading BED files ...")
    gene_chrom, gene_rank = load_bed_files(BED_LABELS, DIR)
    print(f"    {len(gene_chrom):,} genes indexed")

    print("\n[3] Loading anchors files ...")
    pair_evidence, synteny_pairs = load_all_anchors(ANCHORS_CONFIG, DIR)

    # -- 2. Intra-species synteny ----------------------------------------------
    print("\n[4] Detecting intra-species synteny ...")
    intra_synteny = detect_intra_species_synteny(
        kinesin_set, pair_evidence, gene_chrom, gene_rank, ANCHORS_CONFIG
    )
    n_with_partners = sum(1 for g in kinesin_set if g in intra_synteny)
    print(f"    {n_with_partners} kinesin genes have >=1 intra-species partner")
    type_counts = Counter(
        entry["type"]
        for partners in intra_synteny.values()
        for entry in partners
    )
    for t, c in sorted(type_counts.items()):
        print(f"      {t:<12} {c // 2:>4} pairs")

    # -- 3. Duplication classification -----------------------------------------
    print("\n[5] Classifying kinesin-kinesin duplications ...")
    dup_table = build_duplication_table(
        kinesin_set, pair_evidence, gene_chrom, gene_rank, synteny_pairs
    )
    overall = Counter(r["Category"] for r in dup_table)
    print(f"    {len(dup_table)} classified pairs:")
    for cat in CATEGORIES:
        if overall.get(cat, 0):
            print(f"      {cat:<12} {overall[cat]:>4}")

    out_dup = os.path.join(DIR, "kinesin7_duplication_unified.tsv")
    fields = [
        "GeneA", "GeneB", "SpeciesA", "SubA", "SpeciesB", "SubB",
        "ChrA", "ChrB", "Category", "Evidence",
    ]
    with open(out_dup, "w", encoding="utf-8") as fh:
        fh.write("\t".join(fields) + "\n")
        for row in sorted(dup_table, key=lambda r: (r["Category"], r["GeneA"])):
            fh.write("\t".join(str(row.get(f, "")) for f in fields) + "\n")
    print(f"    Written: {out_dup}")

    # -- 4. Ka/Ks --------------------------------------------------------------
    print("\n[6] Loading CDS sequences ...")
    cds_index = load_cds_index(CDS_FILES, kinesin_set)
    print(f"    {len(cds_index)} kinesin CDS sequences indexed")

    if cds_index:
        print("\n[7] Computing Ka/Ks (Nei-Gojobori 1986) ...")
        kaks_rows = compute_kaks(dup_table, cds_index)
        if kaks_rows:
            out_kaks = os.path.join(DIR, "kinesin7_kaks.tsv")
            kaks_fields = fields + ["Ka", "Ks", "Omega"]
            with open(out_kaks, "w", encoding="utf-8") as fh:
                fh.write("\t".join(kaks_fields) + "\n")
                for row in sorted(kaks_rows, key=lambda r: (r["Category"], r["GeneA"])):
                    fh.write("\t".join(str(row.get(f, "")) for f in kaks_fields) + "\n")
            print(f"    Written: {out_kaks}")
    else:
        kaks_rows = []
        print("  [SKIP] Ka/Ks -- no CDS sequences loaded")

    # -- 5. Visualizations -----------------------------------------------------
    print("\n[8] Generating figures ...")
    plot_duplication_summary(dup_table, FIGDIR)
    if kaks_rows:
        plot_kaks_distribution(kaks_rows, FIGDIR)
        plot_kaks_by_category(kaks_rows, FIGDIR)

    print("\n" + "=" * 60)
    print(" Analysis complete.")
    print(f"   Duplication table : kinesin7_duplication_unified.tsv")
    if kaks_rows:
        print(f"   Ka/Ks table       : kinesin7_kaks.tsv")
    print(f"   Figures           : {FIGDIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
