#!/usr/bin/env python
"""
Benchmark XGBoost / LightGBM / PyTorch-MLP across all targets in the
MMnS (medium-Mn steel) CALPHAD dataset.

Targets covered
---------------
  SFE            two-stage: gate classifier (is SFE defined?) + regressor
  Ms             two-stage: gate classifier + regressor
  RA             bounded [0,1] regression (optional logit transform)
  Fm             auto-routed: binary classification or bounded regression
  phases         multi-output regression over the 7 *_Fraction columns
  massfrac:<col> single phase-composition column, filtered to rows where present

Usage
-----
  python steel_ml_benchmark.py --data data/MMnS_dataset_kappa_9-6.parquet \\
      --targets SFE Ms RA Fm phases --models xgb lgbm mlp

  # fast smoke test
  python steel_ml_benchmark.py --data ... --sample 500000 --models xgb

  # a phase-composition target
  python steel_ml_benchmark.py --data ... --targets massfrac:Mass_fraction_C_in_KAPPA_E21

Notes
-----
* Splits are grouped by alloy composition. Never split these rows randomly:
  the same alloy appears at every temperature, so a random split leaks.
* Missingness in SFE/Ms/mass-fraction columns is structural (the phase does
  not exist), not random. It is modelled, never imputed.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

COMP_COLS = [
    "Al_content", "C_content", "Mn_content",
    "Mo_content", "Nb_content", "Si_content", "V_content",
]
BASE_FEATURES = COMP_COLS + ["Temperature"]

PHASE_COLS = [
    "FCC_A1_Fraction", "BCC_A2_Fraction", "CEMENTITE_D011_Fraction",
    "FCC_A1#2_Fraction", "M7C3_D101_Fraction", "M23C6_D84_Fraction",
    "KAPPA_E21_Fraction",
]

SEED = 42


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------

def build_features(df: pd.DataFrame, interactions: bool = True) -> tuple[np.ndarray, list[str]]:
    """Inputs are only composition + temperature. Everything else is an output."""
    feats = {c: df[c].to_numpy(np.float32) for c in BASE_FEATURES}

    if interactions:
        Al, C, Mn = feats["Al_content"], feats["C_content"], feats["Mn_content"]
        Si, Mo, V, Nb = (feats["Si_content"], feats["Mo_content"],
                         feats["V_content"], feats["Nb_content"])
        T = feats["Temperature"]

        # kappa-carbide (Fe,Mn)3AlC formation is driven jointly by Al, Mn, C
        feats["Al_x_C"] = Al * C
        feats["Al_x_Mn"] = Al * Mn
        feats["C_x_Mn"] = C * Mn
        feats["Al_x_Mn_x_C"] = Al * Mn * C
        # austenite stabiliser sum vs. ferrite stabiliser sum (crude but useful)
        feats["gamma_stab"] = Mn + 25.0 * C
        feats["alpha_stab"] = Al + Si + Mo + 5.0 * (V + Nb)
        feats["stab_ratio"] = feats["gamma_stab"] / (feats["alpha_stab"] + 1e-3)
        # carbide formers competing for C
        feats["MC_formers"] = V + Nb + 0.5 * Mo
        feats["C_free_proxy"] = C - 0.2 * feats["MC_formers"]
        # temperature couplings
        feats["T_x_C"] = T * C
        feats["T_x_Al"] = T * Al
        feats["inv_T"] = 1000.0 / T

    names = list(feats.keys())
    X = np.column_stack([feats[n] for n in names]).astype(np.float32)
    return X, names


def composition_group_ids(df: pd.DataFrame) -> np.ndarray:
    """Integer ID per unique alloy composition. Exact + fast on grid data."""
    codes = np.zeros(len(df), dtype=np.int64)
    for c in COMP_COLS:
        _, inv = np.unique(df[c].to_numpy(), return_inverse=True)
        codes = codes * (inv.max() + 1) + inv
    return pd.factorize(codes)[0]


def group_split(groups: np.ndarray, fracs=(0.7, 0.1, 0.2), seed=SEED):
    """Split row indices into train/val/test with no alloy crossing a boundary."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(fracs[0] * n)
    n_va = int(fracs[1] * n)
    sets = (uniq[:n_tr], uniq[n_tr:n_tr + n_va], uniq[n_tr + n_va:])
    return tuple(np.isin(groups, s) for s in sets)


# --------------------------------------------------------------------------
# Target transforms
# --------------------------------------------------------------------------

class Identity:
    name = "identity"
    def fwd(self, y): return y
    def inv(self, z): return z


class Logit:
    """For targets bounded on [0,1] with mass at the endpoints (e.g. RA)."""
    name = "logit"
    def __init__(self, eps=1e-3): self.eps = eps
    def fwd(self, y):
        p = np.clip(y, self.eps, 1 - self.eps)
        return np.log(p / (1 - p))
    def inv(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def reg_metrics(y, p) -> dict:
    err = p - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_err": float(np.max(np.abs(err))),
    }


def clf_metrics(y, prob) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, roc_auc_score)
    pred = (prob >= 0.5).astype(int)
    out = {"n": int(len(y)), "pos_rate": float(y.mean()),
           "acc": float(accuracy_score(y, pred)),
           "bal_acc": float(balanced_accuracy_score(y, pred))}
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, prob))
        out["pr_auc"] = float(average_precision_score(y, prob))
    return out


# --------------------------------------------------------------------------
# Model wrappers  (fit on train, early-stop on val, return predict fn)
# --------------------------------------------------------------------------

@dataclass
class Fitted:
    predict: object
    info: dict = field(default_factory=dict)


def fit_xgb(Xtr, ytr, Xva, yva, task, weight=None, seed=SEED) -> Fitted:
    import xgboost as xgb
    common = dict(
        n_estimators=3000, learning_rate=0.05, max_depth=8,
        min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, tree_method="hist", random_state=seed,
        early_stopping_rounds=50, n_jobs=-1,
    )
    if task == "clf":
        m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                              scale_pos_weight=weight, **common)
    else:
        m = xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    pred = (lambda X: m.predict_proba(X)[:, 1]) if task == "clf" else m.predict
    return Fitted(pred, {"best_iteration": int(m.best_iteration),
                         "importances": m.feature_importances_.tolist()})


def fit_lgbm(Xtr, ytr, Xva, yva, task, weight=None, seed=SEED) -> Fitted:
    import lightgbm as lgb
    common = dict(
        n_estimators=3000, learning_rate=0.05, num_leaves=127,
        min_child_samples=50, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
        n_jobs=-1, verbose=-1,
    )
    if task == "clf":
        m = lgb.LGBMClassifier(objective="binary",
                               scale_pos_weight=weight, **common)
    else:
        m = lgb.LGBMRegressor(objective="l2", **common)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = (lambda X: m.predict_proba(X)[:, 1]) if task == "clf" else m.predict
    return Fitted(pred, {"best_iteration": int(m.best_iteration_),
                         "importances": m.feature_importances_.tolist()})


def fit_mlp(Xtr, ytr, Xva, yva, task, weight=None, seed=SEED,
            hidden=(512, 256, 128), epochs=60, batch=8192, lr=1e-3,
            n_out=1, out_act=None) -> Fitted:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    ymu, ysd = (0.0, 1.0)
    if task == "reg":
        ymu, ysd = float(np.mean(ytr)), float(np.std(ytr)) + 1e-8

    def prep_x(X): return torch.from_numpy(((X - mu) / sd).astype(np.float32))

    layers, d = [], Xtr.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(0.1)]
        d = h
    layers += [nn.Linear(d, n_out)]
    net = nn.Sequential(*layers).to(dev)

    if task == "clf":
        pw = torch.tensor([weight], dtype=torch.float32, device=dev) if weight else None
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        loss_fn = nn.MSELoss()

    ytr_t = torch.from_numpy(
        (ytr if task == "clf" else (ytr - ymu) / ysd).astype(np.float32)
    ).reshape(len(ytr), -1)
    yva_t = torch.from_numpy(
        (yva if task == "clf" else (yva - ymu) / ysd).astype(np.float32)
    ).reshape(len(yva), -1).to(dev)

    ds = torch.utils.data.TensorDataset(prep_x(Xtr), ytr_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                     drop_last=True, num_workers=0)
    Xva_t = prep_x(Xva).to(dev)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

    best, best_state, patience, bad = np.inf, None, 8, 0
    for ep in range(epochs):
        net.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad()
            out = net(xb)
            if out_act == "softmax":
                out = torch.log_softmax(out, -1).exp()
            loss_fn(out, yb).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vo = net(Xva_t)
            if out_act == "softmax":
                vo = torch.softmax(vo, -1)
            vl = float(loss_fn(vo, yva_t))
        sched.step(vl)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()

    def predict(X):
        outs = []
        with torch.no_grad():
            Xt = prep_x(X)
            for i in range(0, len(Xt), 65536):
                o = net(Xt[i:i + 65536].to(dev))
                if task == "clf":
                    o = torch.sigmoid(o)
                elif out_act == "softmax":
                    o = torch.softmax(o, -1)
                else:
                    o = o * ysd + ymu
                outs.append(o.cpu().numpy())
        arr = np.concatenate(outs)
        return arr.ravel() if arr.shape[1] == 1 else arr

    return Fitted(predict, {"epochs_run": ep + 1, "val_loss": best, "device": str(dev)})


FITTERS = {"xgb": fit_xgb, "lgbm": fit_lgbm, "mlp": fit_mlp}


# --------------------------------------------------------------------------
# Task runners
# --------------------------------------------------------------------------

def run_two_stage(name, X, y_raw, masks, model, transform=Identity()):
    """SFE / Ms: gate classifier for 'is it defined', regressor on defined rows."""
    tr, va, te = masks
    defined = ~np.isnan(y_raw)
    res = {"target": name, "model": model, "task": "two_stage"}

    # --- gate ---
    g = defined.astype(np.int32)
    pos = g[tr].mean()
    fit = FITTERS[model](X[tr], g[tr], X[va], g[va], "clf",
                         weight=(1 - pos) / max(pos, 1e-6))
    res["gate"] = clf_metrics(g[te], fit.predict(X[te]))

    # --- regressor on defined rows only ---
    rtr, rva, rte = tr & defined, va & defined, te & defined
    ztr = transform.fwd(y_raw[rtr])
    zva = transform.fwd(y_raw[rva])
    kw = {}
    if model == "mlp":
        kw = dict(hidden=(512, 512, 256, 128), epochs=80)
    fit_r = FITTERS[model](X[rtr], ztr, X[rva], zva, "reg", **kw)
    pred = transform.inv(fit_r.predict(X[rte]))
    res["regressor"] = reg_metrics(y_raw[rte], pred)
    res["regressor"]["transform"] = transform.name
    res["info"] = fit_r.info if model != "mlp" else fit_r.info
    return res


def run_regression(name, X, y_raw, masks, model, transform=Identity()):
    tr, va, te = masks
    ok = ~np.isnan(y_raw)
    tr, va, te = tr & ok, va & ok, te & ok
    kw = dict(hidden=(512, 512, 256, 128), epochs=80) if model == "mlp" else {}
    fit = FITTERS[model](X[tr], transform.fwd(y_raw[tr]), X[va],
                         transform.fwd(y_raw[va]), "reg", **kw)
    pred = transform.inv(fit.predict(X[te]))
    if transform.name == "logit":
        pred = np.clip(pred, 0, 1)
    m = reg_metrics(y_raw[te], pred)
    m["transform"] = transform.name
    return {"target": name, "model": model, "task": "regression",
            "metrics": m, "info": fit.info}


def run_classification(name, X, y_raw, masks, model):
    tr, va, te = masks
    ok = ~np.isnan(y_raw)
    tr, va, te = tr & ok, va & ok, te & ok
    y = (y_raw > 0.5).astype(np.int32)
    pos = y[tr].mean()
    kw = dict(hidden=(256, 256, 128), epochs=60) if model == "mlp" else {}
    fit = FITTERS[model](X[tr], y[tr], X[va], y[va], "clf",
                         weight=(1 - pos) / max(pos, 1e-6), **kw)
    return {"target": name, "model": model, "task": "classification",
            "metrics": clf_metrics(y[te], fit.predict(X[te])), "info": fit.info}


def run_phases(X, Y, masks, model, phase_names):
    """Multi-output. Trees: one model per phase, then renormalise.
       MLP: single softmax head (respects the sum-to-one constraint natively)."""
    tr, va, te = masks
    res = {"target": "phases", "model": model, "task": "multi_regression"}

    if model == "mlp":
        fit = fit_mlp(X[tr], Y[tr], X[va], Y[va], "reg",
                      hidden=(512, 512, 256), epochs=80,
                      n_out=Y.shape[1], out_act="softmax")
        P = fit.predict(X[te])
        res["info"] = fit.info
    else:
        cols = []
        for j in range(Y.shape[1]):
            f = FITTERS[model](X[tr], Y[tr, j], X[va], Y[va, j], "reg")
            cols.append(f.predict(X[te]))
        P = np.clip(np.column_stack(cols), 0, None)
        s = P.sum(1, keepdims=True)
        P = np.divide(P, s, out=np.full_like(P, 1.0 / P.shape[1]), where=s > 1e-9)

    res["per_phase"] = {n: reg_metrics(Y[te, j], P[:, j])
                        for j, n in enumerate(phase_names)}
    res["macro_r2"] = float(np.mean([v["r2"] for v in res["per_phase"].values()]))
    res["mean_abs_sum_err"] = float(np.mean(np.abs(P.sum(1) - Y[te].sum(1))))
    return res


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to the parquet file")
    ap.add_argument("--targets", nargs="+",
                    default=["SFE", "Ms", "RA", "Fm", "phases"])
    ap.add_argument("--models", nargs="+", default=["xgb", "lgbm", "mlp"],
                    choices=["xgb", "lgbm", "mlp"])
    ap.add_argument("--sample", type=int, default=0,
                    help="subsample N whole alloys (0 = use all rows)")
    ap.add_argument("--no-interactions", action="store_true")
    ap.add_argument("--ra-logit", action="store_true",
                    help="model RA on the logit scale instead of raw")
    ap.add_argument("--out", default="benchmark_results.json")
    args = ap.parse_args()

    t0 = time.time()

    # ---- load ----
    need = set(BASE_FEATURES) | set(PHASE_COLS)
    for t in args.targets:
        if t == "phases":
            continue
        need.add(t.split("massfrac:")[-1] if t.startswith("massfrac:") else t)
    print(f"[load] reading {len(need)} columns from {args.data}")
    df = pd.read_parquet(args.data, columns=sorted(need))
    for c in df.columns:  # 5.4 GB -> ~2.7 GB
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    print(f"[load] {df.shape[0]:,} rows x {df.shape[1]} cols  ({time.time()-t0:.1f}s)")

    # ---- groups & split ----
    groups = composition_group_ids(df)
    n_alloys = len(np.unique(groups))
    print(f"[split] {n_alloys:,} unique alloy compositions "
          f"({len(df)/n_alloys:.0f} temperature points each)")

    if args.sample and args.sample < n_alloys:
        rng = np.random.default_rng(SEED)
        keep = rng.choice(np.unique(groups), args.sample, replace=False)
        sel = np.isin(groups, keep)
        df, groups = df[sel].reset_index(drop=True), groups[sel]
        print(f"[split] subsampled to {len(df):,} rows / {args.sample:,} alloys")

    masks = group_split(groups)
    print(f"[split] train {masks[0].sum():,} | val {masks[1].sum():,} | "
          f"test {masks[2].sum():,}  (grouped by composition)")

    X, feat_names = build_features(df, interactions=not args.no_interactions)
    print(f"[feat] {X.shape[1]} features: {', '.join(feat_names)}")

    results = []

    for target in args.targets:
        for model in args.models:
            tag = f"{target} / {model}"
            print(f"\n=== {tag} ===")
            t1 = time.time()
            try:
                if target in ("SFE", "Ms"):
                    r = run_two_stage(target, X, df[target].to_numpy(np.float64),
                                      masks, model)
                    print(f"  gate  : {r['gate']}")
                    print(f"  regr  : {r['regressor']}")

                elif target == "RA":
                    tf = Logit() if args.ra_logit else Identity()
                    r = run_regression("RA", X, df["RA"].to_numpy(np.float64),
                                       masks, model, tf)
                    print(f"  {r['metrics']}")

                elif target == "Fm":
                    y = df["Fm"].to_numpy(np.float64)
                    nuniq = len(np.unique(y[~np.isnan(y)][:2_000_000]))
                    if nuniq <= 2:
                        print(f"  Fm has {nuniq} unique values -> classification")
                        r = run_classification("Fm", X, y, masks, model)
                    else:
                        print(f"  Fm has {nuniq} unique values -> bounded regression")
                        r = run_regression("Fm", X, y, masks, model,
                                           Logit() if args.ra_logit else Identity())
                    print(f"  {r['metrics']}")

                elif target == "phases":
                    Y = df[PHASE_COLS].to_numpy(np.float32)
                    r = run_phases(X, Y, masks, model, PHASE_COLS)
                    for k, v in r["per_phase"].items():
                        print(f"  {k:28s} R2={v['r2']:.4f}  MAE={v['mae']:.5f}")
                    print(f"  macro R2 = {r['macro_r2']:.4f}")

                elif target.startswith("massfrac:"):
                    col = target.split("massfrac:")[-1]
                    y = df[col].to_numpy(np.float64)
                    print(f"  {np.isnan(y).mean()*100:.1f}% missing "
                          f"(phase absent) -> filtered out, not imputed")
                    r = run_regression(col, X, y, masks, model)
                    print(f"  {r['metrics']}")

                else:
                    print(f"  !! unknown target {target}, skipping")
                    continue

                r["seconds"] = round(time.time() - t1, 1)
                r["features"] = feat_names
                results.append(r)
                print(f"  [{r['seconds']}s]")

            except ImportError as e:
                print(f"  !! {model} unavailable: {e}")
            except Exception as e:
                print(f"  !! {tag} failed: {type(e).__name__}: {e}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[done] {len(results)} runs -> {args.out}  ({time.time()-t0:.0f}s total)")

    # ---- leaderboard ----
    rows = []
    for r in results:
        if r["task"] == "two_stage":
            rows.append({"target": r["target"], "model": r["model"],
                         "metric": "R2 (defined rows)", "value": r["regressor"]["r2"],
                         "aux": f"gate AUC {r['gate'].get('roc_auc', float('nan')):.3f}"})
        elif r["task"] == "multi_regression":
            rows.append({"target": r["target"], "model": r["model"],
                         "metric": "macro R2", "value": r["macro_r2"], "aux": ""})
        elif r["task"] == "classification":
            rows.append({"target": r["target"], "model": r["model"],
                         "metric": "ROC AUC",
                         "value": r["metrics"].get("roc_auc", float("nan")),
                         "aux": f"bal_acc {r['metrics']['bal_acc']:.3f}"})
        else:
            rows.append({"target": r["target"], "model": r["model"],
                         "metric": "R2", "value": r["metrics"]["r2"],
                         "aux": f"MAE {r['metrics']['mae']:.4f}"})
    if rows:
        print("\n" + "=" * 72)
        print(pd.DataFrame(rows).pivot_table(
            index=["target", "metric"], columns="model", values="value"
        ).round(4).to_string())


if __name__ == "__main__":
    main()