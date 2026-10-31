from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import infer_tree_depth_from_name, rename_by_aliases, variant_key_frame


TEXT_SUFFIXES = {".csv", ".tsv", ".txt", ".tab", ".gz"}


VARIANT_ALIASES = {
    "sample": ["sample_id", "organ", "leaf", "library", "sample_name"],
    "chrom": ["chr", "chromosome", "contig", "seqid"],
    "pos": ["position", "site", "start"],
    "ref": ["reference", "ref_allele"],
    "alt": ["alternate", "alt_allele", "variant"],
    "ref_depth": ["refdepth", "ref_count", "ref_count_dp", "rd", "n_ref"],
    "alt_depth": ["altdepth", "alt_count", "ad", "n_alt", "variant_depth"],
    "DP": ["dp", "depth", "total_depth", "coverage", "read_depth"],
    "VAF": ["vaf", "af", "allele_frequency", "variant_allele_frequency"],
    "tree": ["tree_id", "individual"],
    "depth_filter": ["dp_filter", "depth", "depth_cutoff", "filter_depth", "cutoff"],
}

ROOT_ALIASES = {
    "sample": ["sample_id", "organ", "leaf", "sample_name"],
    "root2tip_length": ["root_to_tip_length", "root_tip_length", "length", "L", "mean_root2tip_length", "average_root2tip_length"],
    "tree": ["tree_id", "individual"],
}

GENOME_ALIASES = {
    "tree": ["tree_id", "individual"],
    "depth_filter": ["depth", "dp", "dp_filter", "depth_cutoff", "filter_depth", "cutoff"],
    "effective_genome_size": ["genome_size", "callable_genome_size", "effective_callable_genome_size", "G"],
}


@dataclass(frozen=True)
class LoadedTable:
    name: str
    frame: pd.DataFrame


def _looks_like_text(name: str) -> bool:
    p = Path(name)
    if p.name.startswith("."):
        return False
    if p.suffix.lower() in TEXT_SUFFIXES:
        return True
    if p.suffix.lower() == ".gz" and p.with_suffix("").suffix.lower() in TEXT_SUFFIXES:
        return True
    return False


def read_table_auto(source, name: str | None = None) -> pd.DataFrame:
    """Read a delimited table from a path, bytes buffer, or file-like object."""
    if name is None:
        name = str(source)
    lower = name.lower()
    kwargs = {}
    if lower.endswith(".csv") or lower.endswith(".csv.gz"):
        kwargs["sep"] = ","
    else:
        kwargs["sep"] = None
        kwargs["engine"] = "python"
    try:
        return pd.read_csv(source, **kwargs)
    except Exception:
        # Fallback for whitespace-delimited text with no obvious separator.
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, sep=r"\s+", engine="python")


def load_tables(path: str | Path) -> list[LoadedTable]:
    """Load one or more delimited tables from a file, directory, or zip archive."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    tables: list[LoadedTable] = []
    if p.is_dir():
        for child in sorted(p.rglob("*")):
            if child.is_file() and _looks_like_text(child.name):
                tables.append(LoadedTable(str(child), read_table_auto(child)))
    elif p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            for name in sorted(zf.namelist()):
                if name.endswith("/") or not _looks_like_text(name):
                    continue
                with zf.open(name, "r") as fh:
                    data = io.BytesIO(fh.read())
                    tables.append(LoadedTable(name, read_table_auto(data, name=name)))
    else:
        tables.append(LoadedTable(str(p), read_table_auto(p)))

    if not tables:
        raise ValueError(f"No readable delimited tables found in {path}")
    return tables


def read_variants(path: str | Path, dataset_label: str, tree_label: str | None = None) -> pd.DataFrame:
    """Read DP-tier variant tables. The tree is taken from a 'tree' column, else from `tree_label`
    (config input.tree), else from a 'treeNN' token in the file name."""
    frames = []
    for tab in load_tables(path):
        df = rename_by_aliases(tab.frame, VARIANT_ALIASES)
        inferred_tree, inferred_depth = infer_tree_depth_from_name(tab.name)
        if tree_label:
            inferred_tree = str(tree_label)
        if "tree" not in df.columns or df["tree"].isna().all():
            if inferred_tree is None:
                raise ValueError(f"Could not infer tree from file name and no 'tree' column exists: {tab.name} "
                                 "(set input.tree in the config)")
            df["tree"] = inferred_tree
        if "depth_filter" not in df.columns or df["depth_filter"].isna().all():
            if inferred_depth is None:
                raise ValueError(
                    f"Could not infer depth_filter from file name and no depth_filter column exists: {tab.name}"
                )
            df["depth_filter"] = inferred_depth
        df["source_file"] = tab.name
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    required = ["sample", "chrom", "pos", "ref", "alt", "ref_depth", "alt_depth"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Variant table is missing required columns: {missing}")

    for c in ["ref_depth", "alt_depth"]:
        out[c] = pd.to_numeric(out[c], errors="raise").astype(int)
    if "DP" not in out.columns:
        out["DP"] = out["ref_depth"] + out["alt_depth"]
    else:
        out["DP"] = pd.to_numeric(out["DP"], errors="raise").astype(int)
    if "VAF" not in out.columns:
        out["VAF"] = np.where(out["DP"] > 0, out["alt_depth"] / out["DP"], np.nan)
    else:
        out["VAF"] = pd.to_numeric(out["VAF"], errors="coerce")

    out["tree"] = out["tree"].astype(str)
    out["depth_filter"] = pd.to_numeric(out["depth_filter"], errors="raise").astype(int)
    out["dataset_label"] = dataset_label
    out["is_positive_observation"] = out["alt_depth"] > 0
    out["variant_id"] = variant_key_frame(out)
    return out


def read_root2tip(path: str | Path) -> pd.DataFrame:
    tables = load_tables(path)
    if len(tables) != 1:
        # Multiple files are allowed, but are simply concatenated.
        df = pd.concat([rename_by_aliases(t.frame, ROOT_ALIASES) for t in tables], ignore_index=True)
    else:
        df = rename_by_aliases(tables[0].frame, ROOT_ALIASES)
    if "root2tip_length" not in df.columns:
        raise ValueError("Root-to-tip table must contain root2tip_length or an accepted alias.")
    df["root2tip_length"] = pd.to_numeric(df["root2tip_length"], errors="raise")
    if "sample" in df.columns:
        df["sample"] = df["sample"].astype(str)
    if "tree" in df.columns:
        df["tree"] = df["tree"].astype(str)
    return df


def read_effective_genome(path: str | Path, tree_label: str | None = None) -> pd.DataFrame:
    frames = []
    for tab in load_tables(path):
        df = rename_by_aliases(tab.frame, GENOME_ALIASES)
        inferred_tree, inferred_depth = infer_tree_depth_from_name(tab.name)
        if tree_label:
            inferred_tree = str(tree_label)
        if "tree" not in df.columns or df["tree"].isna().all():
            if inferred_tree is not None:
                df["tree"] = inferred_tree
        if "depth_filter" not in df.columns or df["depth_filter"].isna().all():
            if inferred_depth is not None:
                df["depth_filter"] = inferred_depth
        df["source_file"] = tab.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    required = ["tree", "depth_filter", "effective_genome_size"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Effective genome-size table is missing required columns: {missing}")
    out["tree"] = out["tree"].astype(str)
    out["depth_filter"] = pd.to_numeric(out["depth_filter"], errors="raise").astype(int)
    out["effective_genome_size"] = pd.to_numeric(out["effective_genome_size"], errors="raise")
    out = out[["tree", "depth_filter", "effective_genome_size"]].drop_duplicates()
    if out.duplicated(["tree", "depth_filter"]).any():
        dup = out[out.duplicated(["tree", "depth_filter"], keep=False)]
        raise ValueError(f"Duplicate effective genome-size entries for tree/depth:\n{dup}")
    return out


def root_length_for_tree_samples(root: pd.DataFrame, tree: str, samples: Iterable[str]) -> tuple[float, list[str], int, bool]:
    """Return total root-to-tip length, missing sample names, n root samples considered, and sample-specific flag."""
    samples = sorted({str(s) for s in samples})
    r = root.copy()
    if "tree" in r.columns:
        rt = r[r["tree"].astype(str) == str(tree)].copy()
        if rt.empty:
            rt = r.copy()  # permit global table if tree labels mismatch, validation will expose this elsewhere
    else:
        rt = r

    sample_specific = "sample" in rt.columns and not rt["sample"].isna().all()
    if sample_specific:
        rt = rt.copy()
        rt["sample"] = rt["sample"].astype(str)
        matched = rt[rt["sample"].isin(samples)]
        missing = sorted(set(samples) - set(matched["sample"].astype(str)))
        if not missing and not matched.empty:
            return float(matched.drop_duplicates("sample")["root2tip_length"].sum()), [], int(matched["sample"].nunique()), True
        # If sample-specific matching fails completely, treat as average fallback below.
        if len(missing) < len(samples):
            return float(matched.drop_duplicates("sample")["root2tip_length"].sum()), missing, int(matched["sample"].nunique()), True

    # Average-length fallback: one or more rows without sample mapping.
    mean_len = float(rt["root2tip_length"].mean())
    return mean_len * len(samples), [], int(len(rt)), False


def root_lengths_by_sample(root: pd.DataFrame, tree: str, samples: Iterable[str]) -> tuple[dict[str, float], bool]:
    """Return {sample: root2tip_length} for the requested samples, plus a sample_specific flag.

    Used for the depth-window metric and written to the per-unit outputs, which need each organ's own
    L_s rather than a single value pooled/summed across organs. Falls back to
    the tree (or global) average length for every requested sample when no
    sample-specific mapping is available, mirroring root_length_for_tree_samples.
    """
    samples = sorted({str(s) for s in samples})
    r = root.copy()
    if "tree" in r.columns:
        rt = r[r["tree"].astype(str) == str(tree)].copy()
        if rt.empty:
            rt = r.copy()
    else:
        rt = r

    sample_specific = "sample" in rt.columns and not rt["sample"].isna().all()
    if sample_specific:
        rt = rt.copy()
        rt["sample"] = rt["sample"].astype(str)
        matched = rt.drop_duplicates("sample").set_index("sample")["root2tip_length"]
        out = {s: float(matched[s]) for s in samples if s in matched.index}
        if len(out) == len(samples):
            return out, True
        if len(out) > 0:
            # Partial sample-specific match: fill any missing samples with the
            # mean of the samples we do have, so every requested sample still
            # gets an entry (an organ-level average won't silently drop).
            mean_len = float(np.mean(list(out.values())))
            for s in samples:
                out.setdefault(s, mean_len)
            return out, True

    mean_len = float(rt["root2tip_length"].mean())
    return {s: mean_len for s in samples}, False
