"""Regenerate the synthetic example data in vafSOMA_inputs/examples/example_tree/ with simSOMA.

Requires simSOMA >= 0.2.3 (`pip install git+https://github.com/jlab-code/simSOMA`). The simulation settings are in
vafSOMA_inputs/examples/example_tree_simsoma_layers.json (simSOMA layered template; per-layer rates raised five-fold so
that the small example tree carries enough fixed mutations). Usage, from the repository root:

    python vafSOMA_scripts/make_example_data.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "vafSOMA_inputs" / "examples"
OUT = EX / "example_tree"
NOMINAL_GENOME = 200_000_000        # nominal effective genome size (simSOMA rates are per genome and time unit)


def main() -> int:
    cfg = json.loads((EX / "example_tree_simsoma_layers.json").read_text())
    topo_name = cfg["topology"]["topology_json"]
    topo = json.loads((EX / topo_name).read_text())
    with tempfile.TemporaryDirectory() as tmp:
        cfg["topology"]["topology_json"] = str(EX / cfg["topology"]["topology_json"])
        cfg["output"]["dir"] = str(Path(tmp) / "sim")
        (Path(tmp) / "layers.json").write_text(json.dumps(cfg))
        subprocess.run([sys.executable, "-m", "simsoma.cli", "layers", str(Path(tmp) / "layers.json")], check=True)
        rep = Path(tmp) / "sim" / "replicate_0000"
        if OUT.exists():
            shutil.rmtree(OUT)
        OUT.mkdir(parents=True)
        tables = sorted(rep.glob("vafsoma_dp_*_vaf.csv"), key=lambda p: int(p.stem.split("_")[2]))
        with zipfile.ZipFile(OUT / "vaf_tables.zip", "w", zipfile.ZIP_DEFLATED) as zf:
            for t in tables:
                zf.write(t, t.name)
        cuts = [int(t.stem.split("_")[2]) for t in tables]
        pd.DataFrame({"depth_filter": cuts, "effective_genome_size": NOMINAL_GENOME}).to_csv(
            OUT / "effective_genome_size.tsv", sep="\t", index=False)
        organs = [e for e in topo["events"] if e["type"] == "organ"]
        pd.DataFrame({"sample": [e["organ"] for e in organs], "root2tip_length": [e["age"] for e in organs]}).to_csv(
            OUT / "root2tip.tsv", sep="\t", index=False)
        pd.DataFrame({"tree": "example", "sample": [e["organ"] for e in organs],
                      "group": [e["branch"] for e in organs]}).to_csv(OUT / "replicate_groups.tsv", sep="\t", index=False)
        shutil.copy(rep / "layer_carriers.csv.gz", OUT / "truth_layer_carriers.csv.gz")
        rec = json.loads((Path(tmp) / "sim" / "layered_settings.json").read_text())
        rec.get("config", {}).get("topology", {})["topology_json"] = f"../{topo_name}"      # no local paths
        rec.get("config", {}).get("output", {})["dir"] = "(temporary)"
        (OUT / "simsoma_layered_settings.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
