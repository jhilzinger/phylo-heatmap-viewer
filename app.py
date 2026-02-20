"""
Phylogenetic Tree + Phenotype Heatmap Viewer
Uses Biopython Phylo for matplotlib-native tree drawing (PDF-exportable),
and toytree for tip-order extraction (more accurate layout algorithm).
"""

import hashlib
import io
import os
import re
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ── Biopython ──────────────────────────────────────────────────────────────
try:
    from Bio import SeqIO, AlignIO, Phylo
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
    HAS_BIO = True
except ImportError:
    HAS_BIO = False
    st.error("Biopython is not installed.  Run: pip install biopython")
    st.stop()

# ── toytree (for tip-order only) ───────────────────────────────────────────
try:
    import toytree
    import toyplot.svg
    HAS_TOY = True
except ImportError:
    HAS_TOY = False

# ── tool discovery ─────────────────────────────────────────────────────────
FASTTREE_CANDIDATES = [
    "FastTree", "fasttree", "FastTree2", "VeryFastTree",
    "/opt/homebrew/bin/VeryFastTree",
    "/usr/local/bin/FastTree",
    "/usr/bin/FastTree",
]
MAFFT_CANDIDATES = ["mafft", "/usr/local/bin/mafft"]


def _probe_exe(cmd: list[str]) -> bool:
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=5)
        return True          # any returncode is fine; just need it not to raise
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        return False


def _find_exe(candidates: list[str], probe_flag: str) -> str | None:
    for c in candidates:
        if _probe_exe([c, probe_flag]):
            return c
    return None


MAFFT_EXE    = _find_exe(MAFFT_CANDIDATES,    "--help")
FASTTREE_EXE = _find_exe(FASTTREE_CANDIDATES, "-help")

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phylo + Phenotype Viewer",
    layout="wide",
    page_icon="🌿",
)
st.title("🌿 Phylogenetic Tree & Phenotype Heatmap")

with st.expander("⚙️ Tool availability", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        if MAFFT_EXE:
            st.success(f"MAFFT found: `{MAFFT_EXE}`")
        else:
            st.warning("MAFFT **not found** — NJ fallback will be used")
    with c2:
        if FASTTREE_EXE:
            st.success(f"FastTree found: `{FASTTREE_EXE}`")
        else:
            st.warning("FastTree **not found** — Biopython NJ fallback")
    if not MAFFT_EXE or not FASTTREE_EXE:
        st.code("conda install -c bioconda mafft fasttree\n# or on macOS:\nbrew install mafft veryfasttree")

# ── sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader(
        "Upload CSV", type=["csv"],
        help="Needs: AccessionID, AAseq, PA, X3HB, GA, X5HFC, HEX",
    )
    default_csv = Path("20260202_PhylogeneticScreen_ScoresAndSequence_AAs_Arvind.csv")

    cmap_choice = st.selectbox(
        "Heatmap colormap",
        ["viridis", "plasma", "inferno", "magma", "cividis",
         "coolwarm", "RdYlGn", "YlOrRd", "Blues"],
    )
    show_scale = st.checkbox("Show branch-length scale bar", value=True)
    midpoint   = st.checkbox("Midpoint-root the tree", value=False)

    st.markdown("---")
    st.subheader("Export")
    dl_newick = st.button("⬇ Download Newick")
    dl_pdf    = st.button("⬇ Download figure as PDF")


# ── constants ──────────────────────────────────────────────────────────────
PHENO_COLS   = ["PA", "X3HB", "GA", "X5HFC", "HEX"]
PHENO_LABELS = ["PA", "3HB", "GA", "5HFC", "HEX"]
VALID_AAS    = set("ACDEFGHIKLMNPQRSTVWY")


def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# ── data loading ───────────────────────────────────────────────────────────
def load_data(src) -> pd.DataFrame:
    if src is None:
        if default_csv.exists():
            df = pd.read_csv(default_csv)
        else:
            st.error(
                f"Default file `{default_csv}` not found and no file uploaded.\n"
                "Please upload a CSV using the sidebar."
            )
            st.stop()
    else:
        df = pd.read_csv(src)

    required = {"AccessionID", "AAseq"} | set(PHENO_COLS)
    missing  = required - set(df.columns)
    if missing:
        st.error(f"CSV is missing required columns: {missing}")
        st.stop()

    df = df.dropna(subset=["AccessionID", "AAseq"]).reset_index(drop=True)
    return df


def clean_sequences(df: pd.DataFrame) -> list[SeqRecord]:
    records, warned = [], []
    for _, row in df.iterrows():
        seq_raw = str(row["AAseq"]).upper().replace("-", "").replace("*", "").strip()
        bad = set(seq_raw) - VALID_AAS
        if bad:
            warned.append(f"{row['AccessionID']}: {bad} removed")
            seq_raw = "".join(c for c in seq_raw if c in VALID_AAS)
        if len(seq_raw) < 10:
            continue
        records.append(
            SeqRecord(Seq(seq_raw), id=str(row["AccessionID"]), description="")
        )
    if warned:
        with st.expander(f"⚠️ {len(warned)} sequences had non-standard AA characters (stripped)"):
            st.write("\n".join(warned))
    return records


# ── Step 1: MSA ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_msa(fasta_bytes: bytes, _hash: str) -> tuple[str, str]:
    """Returns (aligned_fasta_str, method_name)."""
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "input.fasta")
        with open(inp, "wb") as f:
            f.write(fasta_bytes)

        # MAFFT
        if MAFFT_EXE:
            try:
                result = subprocess.run(
                    [MAFFT_EXE, "--auto", "--thread", "-1", inp],
                    capture_output=True, timeout=600,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.decode("utf-8", errors="replace"), "MAFFT"
                stderr = result.stderr.decode("utf-8", errors="replace")
                st.warning(f"MAFFT failed (rc={result.returncode}): {stderr[:400]}")
            except subprocess.TimeoutExpired:
                st.warning("MAFFT timed out — trying MUSCLE fallback")
            except Exception as e:
                st.warning(f"MAFFT error: {e}")

        # MUSCLE fallback
        muscle_exe = _find_exe(["muscle"], "-help")
        if muscle_exe:
            out = os.path.join(tmp, "aligned.fasta")
            try:
                result = subprocess.run(
                    [muscle_exe, "-in", inp, "-out", out],
                    capture_output=True, timeout=600,
                )
                if result.returncode == 0 and os.path.exists(out):
                    return open(out).read(), "MUSCLE"
            except Exception as e:
                st.warning(f"MUSCLE error: {e}")

        # Pass-through unaligned — Biopython NJ handles unaligned via identity matrix
        return fasta_bytes.decode("utf-8", errors="replace"), "none (unaligned)"


# ── Step 2: Tree ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_tree(aligned_fasta: str, msa_method: str, _hash: str) -> tuple[str, str]:
    """Returns (newick_str, method_name)."""
    with tempfile.TemporaryDirectory() as tmp:
        aln_file = os.path.join(tmp, "aligned.fasta")
        with open(aln_file, "w") as f:
            f.write(aligned_fasta)

        # FastTree (only if we have a real alignment)
        if FASTTREE_EXE and msa_method not in ("none (unaligned)", ""):
            try:
                result = subprocess.run(
                    [FASTTREE_EXE, "-lg", "-gamma", aln_file],
                    capture_output=True, timeout=600,
                )
                if result.returncode == 0 and result.stdout:
                    nwk = result.stdout.decode("utf-8", errors="replace").strip()
                    if nwk.startswith("("):
                        return nwk, f"FastTree ({Path(FASTTREE_EXE).name})"
                stderr = result.stderr.decode("utf-8", errors="replace")
                st.warning(f"FastTree failed (rc={result.returncode}): {stderr[:400]}")
            except subprocess.TimeoutExpired:
                st.warning("FastTree timed out — falling back to Biopython NJ")
            except Exception as e:
                st.warning(f"FastTree error: {e}")

        # Biopython NJ fallback
        try:
            alignment = AlignIO.read(io.StringIO(aligned_fasta), "fasta")
            calculator  = DistanceCalculator("identity")
            constructor = DistanceTreeConstructor(calculator, "nj")
            tree        = constructor.build_tree(alignment)
            buf = io.StringIO()
            Phylo.write(tree, buf, "newick")
            return buf.getvalue().strip(), "Biopython NJ (identity matrix)"
        except Exception as e:
            st.error(f"Biopython NJ tree failed: {e}")
            st.stop()


# ── tip order ─────────────────────────────────────────────────────────────
def get_tip_order(newick_str: str, do_midpoint: bool) -> list[str]:
    """
    Returns tip labels in top-to-bottom display order.
    Uses toytree if available (more accurate layout); falls back to Biopython.
    """
    if HAS_TOY:
        tree = toytree.tree(newick_str)
        if do_midpoint:
            tree = tree.mod.root_on_midpoint()
        # get_tip_labels() returns tips in bottom-up y order (idx 0 = bottom).
        # Reverse for top-to-bottom.
        return list(reversed(tree.get_tip_labels()))
    else:
        # Biopython get_terminals() returns top-to-bottom order after draw
        bio_tree = Phylo.read(io.StringIO(newick_str), "newick")
        if do_midpoint:
            bio_tree.root_at_midpoint()
        return [t.name for t in bio_tree.get_terminals()]


# ── Step 3: Figure ───────────────────────────────────────────────────────────
def build_figure(
    newick_str: str,
    df: pd.DataFrame,
    tip_order: list[str],
    cmap: str,
    show_scalebar: bool,
    do_midpoint: bool,
    tree_method: str,
    msa_method: str,
) -> plt.Figure:
    """
    Left panel : Biopython Phylo phylogram (matplotlib-native, PDF-safe)
    Right panel: seaborn/imshow heatmap, rows = tip_order (top→bottom)
    """
    bio_tree = Phylo.read(io.StringIO(newick_str), "newick")
    if do_midpoint:
        bio_tree.root_at_midpoint()

    n_tips = len(tip_order)
    fig_h  = max(6, n_tips * 0.30 + 2)

    fig = plt.figure(figsize=(18, fig_h))
    gs  = gridspec.GridSpec(
        1, 3,
        figure=fig,
        width_ratios=[3, 1, 0.07],
        wspace=0.04,
        left=0.01, right=0.97,
        top=0.93,  bottom=0.07,
    )
    ax_tree = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])
    ax_cbar = fig.add_subplot(gs[2])

    # ── tree panel ──────────────────────────────────────────────────────
    Phylo.draw(
        bio_tree,
        axes=ax_tree,
        do_show=False,
        branch_labels=None,
        label_colors=None,
    )
    ax_tree.set_xlabel("Branch length" if show_scalebar else "", fontsize=8)
    if not show_scalebar:
        ax_tree.xaxis.set_visible(False)

    # tighten margins so tips align with heatmap rows
    ax_tree.set_ylim(n_tips + 0.5, 0.5)   # top→bottom: 1..n_tips

    title_str = (
        f"Phylogram  ·  MSA: {msa_method}  ·  Tree: {tree_method}"
        + ("  ·  midpoint-rooted" if do_midpoint else "")
    )
    ax_tree.set_title(title_str, fontsize=8, pad=4)
    ax_tree.tick_params(axis="y", labelsize=max(5, min(9, 200 // n_tips)))

    # ── heatmap panel ───────────────────────────────────────────────────
    df_idx  = df.set_index("AccessionID")
    present = [t for t in tip_order if t in df_idx.index]
    missing = [t for t in tip_order if t not in df_idx.index]

    if missing:
        st.warning(
            f"{len(missing)} tree tips not found in CSV (shown as NaN): "
            + ", ".join(missing[:8]) + ("…" if len(missing) > 8 else "")
        )

    heat_df = df_idx.reindex(tip_order)[PHENO_COLS].copy().astype(float)

    # column-wise min-max normalization
    for col in PHENO_COLS:
        mn, mx = heat_df[col].min(), heat_df[col].max()
        heat_df[col] = (heat_df[col] - mn) / (mx - mn) if mx > mn else 0.5
    heat_df.columns = PHENO_LABELS

    im = ax_heat.imshow(
        heat_df.values,
        aspect="auto",
        cmap=cmap,
        vmin=0, vmax=1,
        interpolation="nearest",
        origin="upper",
    )
    ax_heat.set_xticks(range(len(PHENO_LABELS)))
    ax_heat.set_xticklabels(PHENO_LABELS, rotation=45, ha="right", fontsize=8)
    ax_heat.set_yticks([])
    ax_heat.set_title("Phenotype\n(col-normalised)", fontsize=8, pad=4)

    # align y-extents of heatmap with tree
    # imshow rows go 0..n-1, tree y goes 1..n
    ax_heat.set_ylim(n_tips - 0.5, -0.5)  # flipped so row 0 is top

    # colorbar
    cb = fig.colorbar(im, cax=ax_cbar)
    cb.set_label("Normalized score", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    fig.suptitle(
        "Phylogenetic Tree & Phenotype Heatmap",
        fontsize=12, fontweight="bold", y=0.97,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

# 1. Load data
raw_bytes = uploaded.read() if uploaded else None
df        = load_data(io.BytesIO(raw_bytes) if raw_bytes else None)

try:
    hash_src = raw_bytes if raw_bytes else default_csv.read_bytes()
except Exception:
    hash_src = b""
content_hash = file_hash(hash_src)

st.info(
    f"Loaded **{len(df)}** sequences from "
    f"{'uploaded file' if uploaded else f'`{default_csv.name}`'}."
)

# 2. Build FASTA
records     = clean_sequences(df)
fasta_bytes = ("\n".join(f">{r.id}\n{str(r.seq)}" for r in records)).encode()

# 3. MSA
st.subheader("Step 1 — Multiple Sequence Alignment")
with st.spinner("Running MSA…"):
    aligned_fasta, msa_method = run_msa(fasta_bytes, content_hash)
st.success(f"MSA complete — method: **{msa_method}**")

# 4. Tree
st.subheader("Step 2 — Phylogenetic Tree Construction")
with st.spinner("Building tree…"):
    newick_str, tree_method = build_tree(aligned_fasta, msa_method, content_hash)
st.success(f"Tree built — method: **{tree_method}**")

with st.expander("Raw Newick string"):
    st.code(
        newick_str[:3000] + ("…" if len(newick_str) > 3000 else ""),
        language="text",
    )

# 5. Tip order
tip_order = get_tip_order(newick_str, midpoint)

# 6. Visualize
st.subheader("Step 3 — Visualization")
tip_source = "toytree" if HAS_TOY else "Biopython Phylo"
st.caption(f"Tip order from **{tip_source}** · {len(tip_order)} tips")

with st.spinner("Rendering figure…"):
    fig = build_figure(
        newick_str, df, tip_order,
        cmap_choice, show_scale, midpoint,
        tree_method, msa_method,
    )

st.pyplot(fig, use_container_width=True)

# 7. Downloads
if dl_newick:
    st.download_button(
        "💾 Save tree.nwk",
        data=newick_str,
        file_name="tree.nwk",
        mime="text/plain",
    )

if dl_pdf:
    pdf_buf = io.BytesIO()
    fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
    pdf_buf.seek(0)
    st.download_button(
        "💾 Save figure.pdf",
        data=pdf_buf,
        file_name="phylo_heatmap.pdf",
        mime="application/pdf",
    )
