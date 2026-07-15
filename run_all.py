#!/usr/bin/env python
"""
Run the full experiment sweep in resumable stages.

Each stage is a separate subprocess: the parquet reloads per stage (~1-2 min),
but a crash in stage 4 doesn't cost you stages 1-3. Completed stages are
skipped on re-run, so you can Ctrl-C and restart freely.

  python run_all.py --data data/MMnS_dataset_kappa_9-6.parquet
  python run_all.py --data ... --sample 3000        # fast full-coverage pass
  python run_all.py --data ... --stages main        # skip the massfrac sweep
  python run_all.py --data ... --force              # ignore existing results
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ELEMENTS = ["Fe", "C", "Mn", "Si", "Al", "Mo", "Nb", "V"]
PHASES = ["FCC_A1", "BCC_A2", "CEMENTITE_D011", "FCC_A1#2",
          "M7C3_D101", "M23C6_D84", "KAPPA_E21"]

# Phases whose composition columns are worth modelling. M23C6 is 99.8% missing
# (~17k rows of 9.5M) -- it is listed but excluded by default; a model there
# would be fitting noise on a handful of alloys.
DENSE_PHASES = ["FCC_A1", "BCC_A2", "FCC_A1#2", "KAPPA_E21",
                "CEMENTITE_D011", "M7C3_D101"]

MODELS = ["xgb", "lgbm", "mlp"]

MAIN_TARGETS = ["SFE", "Ms", "RA", "Fm", "phases"]


def massfrac_targets(phases) -> list[str]:
    return [f"massfrac:Mass_fraction_{e}_in_{p}" for p in phases for e in ELEMENTS]


def build_stages(include_sparse: bool) -> dict[str, list[str]]:
    """Stage name -> target list. Ordered cheapest/most-valuable first."""
    phases = PHASES if include_sparse else DENSE_PHASES
    stages = {
        "main": MAIN_TARGETS,
    }
    # one stage per phase keeps each subprocess to ~8 targets
    for p in phases:
        key = "mf_" + p.replace("#", "").replace("_", "").lower()
        stages[key] = massfrac_targets([p])
    return stages


def run_stage(name, targets, args) -> bool:
    out = Path(args.outdir) / f"{name}.json"
    if out.exists() and not args.force:
        n = len(json.loads(out.read_text()))
        print(f"[skip] {name}: {out} exists ({n} runs). --force to redo.")
        return True

    cmd = [sys.executable, args.script,
           "--data", args.data,
           "--targets", *targets,
           "--models", *args.models,
           "--out", str(out)]
    if args.sample:
        cmd += ["--sample", str(args.sample)]
    if args.no_interactions:
        cmd += ["--no-interactions"]
    if args.ra_logit:
        cmd += ["--ra-logit"]

    log = Path(args.outdir) / f"{name}.log"
    print(f"\n{'='*72}\n[stage] {name}  ({len(targets)} targets x "
          f"{len(args.models)} models)\n[log]   {log}\n{'='*72}")

    t0 = time.time()
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        proc.wait()

    dt = time.time() - t0
    ok = proc.returncode == 0
    print(f"[stage] {name} {'OK' if ok else 'FAILED rc=' + str(proc.returncode)} "
          f"in {dt/60:.1f} min")
    return ok


def collate(outdir: Path):
    """Merge every stage's json into one flat leaderboard."""
    import pandas as pd

    rows = []
    for f in sorted(outdir.glob("*.json")):
        if f.name == "ALL_RESULTS.json":
            continue
        for r in json.loads(f.read_text()):
            base = {"stage": f.stem, "target": r["target"],
                    "model": r["model"], "task": r["task"],
                    "seconds": r.get("seconds")}
            if r["task"] == "two_stage":
                base |= {"score": r["regressor"]["r2"], "metric": "R2_defined",
                         "mae": r["regressor"]["mae"], "n_test": r["regressor"]["n"],
                         "gate_auc": r["gate"].get("roc_auc")}
            elif r["task"] == "multi_regression":
                base |= {"score": r["macro_r2"], "metric": "macro_R2"}
            elif r["task"] == "classification":
                base |= {"score": r["metrics"].get("roc_auc"), "metric": "ROC_AUC",
                         "n_test": r["metrics"]["n"]}
            else:
                base |= {"score": r["metrics"]["r2"], "metric": "R2",
                         "mae": r["metrics"]["mae"], "n_test": r["metrics"]["n"]}
            rows.append(base)

    if not rows:
        print("[collate] nothing to collate")
        return

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "leaderboard.csv", index=False)

    print(f"\n{'='*72}\nLEADERBOARD  ({len(df)} runs)\n{'='*72}")
    pivot = df.pivot_table(index=["target", "metric"], columns="model",
                           values="score").round(4)
    print(pivot.to_string())

    print(f"\nBest model per target:")
    best = df.loc[df.groupby("target")["score"].idxmax()]
    print(best[["target", "model", "metric", "score"]].to_string(index=False))
    print(f"\n[collate] -> {outdir/'leaderboard.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--script", default="steel_ml_benchmark.py")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--stages", nargs="+", default=["all"],
                    help="'all', 'main', 'massfrac', or explicit stage names")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--include-sparse", action="store_true",
                    help="also model M23C6 columns (99.8%% missing)")
    ap.add_argument("--no-interactions", action="store_true")
    ap.add_argument("--ra-logit", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--collate-only", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.collate_only:
        collate(outdir)
        return

    all_stages = build_stages(args.include_sparse)

    if args.stages == ["all"]:
        chosen = list(all_stages)
    elif args.stages == ["massfrac"]:
        chosen = [s for s in all_stages if s != "main"]
    else:
        chosen = []
        for s in args.stages:
            if s in all_stages:
                chosen.append(s)
            else:
                sys.exit(f"unknown stage '{s}'. options: {', '.join(all_stages)}")

    print(f"[plan] {len(chosen)} stages: {', '.join(chosen)}")
    print(f"[plan] models: {', '.join(args.models)}")
    print(f"[plan] sample: {args.sample or 'ALL alloys'}")

    t0 = time.time()
    failed = []
    for name in chosen:
        if not run_stage(name, all_stages[name], args):
            failed.append(name)

    print(f"\n[all] finished in {(time.time()-t0)/60:.1f} min")
    if failed:
        print(f"[all] FAILED stages: {', '.join(failed)} (see logs; re-run to retry)")

    collate(outdir)


if __name__ == "__main__":
    main()