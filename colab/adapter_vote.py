# -*- coding: utf-8 -*-
"""Row-wise reliability adapters for the routed TabM model.

This is deliberately *not* the old R/F two-head experiment.  That experiment
asked one shared model to replace the regular/futures specialists and lost
badly.  Here every routed branch keeps its own model.  Within each branch a
normal head is the anchor and four small output adapters are trained for
row-local regimes:

* low prior pitcher sample size;
* low prior batter sample size;
* handedness matchup;
* two-strike count.

The vote is a fixed, row-local convex logit vote.  It uses only columns in that
row: no test-row aggregation, test distribution estimate, or current-pitch
TrackMan information.  The count features restore the three `asof_*_n`
columns omitted by F44, together with evidence-weighted versions of signals
already present in F44.  They are evaluated as a separate arm before the
adapter conclusion is made.

Protocol
--------
* `base44`: exact all / regular / futures routed control;
* `counts`: same architecture with six count/evidence features;
* `adapt`: counts plus a shared backbone and five output heads.

The default is a one-seed, three-season screen.  A positive screen must be
rerun with three seeds before it can become a candidate.  Scores are reported
on the complete R+F fold, not a hand-picked slice.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as nnf


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

DATA = Path(os.environ.get("AIMERS_DATA", "/root/aimers/open (1)/data"))
OUT = Path(os.environ.get("AIMERS_OUT", "/root/adapter_vote_out"))
OUT.mkdir(parents=True, exist_ok=True)
FOLDS = tuple(int(x) for x in os.environ.get("AV_FOLDS", "2022,2023,2024").split(","))
SEEDS = tuple(int(x) for x in os.environ.get("AV_SEEDS", "42").split(","))
BS = int(os.environ.get("AV_BS", "2048"))
K = int(os.environ.get("AV_K", "32"))
DBLOCK = int(os.environ.get("AV_DBLOCK", "256"))
ADAPTER_LOSS = float(os.environ.get("AV_ADAPTER_LOSS", "0.35"))
ADAPTER_ANCHOR = float(os.environ.get("AV_ADAPTER_ANCHOR", "1.0"))
RUN_ADAPTERS = os.environ.get("AV_RUN_ADAPTERS", "1") != "0"
FEATURE_MODE = os.environ.get("AV_FEATURE_MODE", "all6")
OLD_F_MAX, OLD_F_WEIGHT = 2022, 0.10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda" and os.environ.get("AV_AMP", "1") != "0"

import features44 as FF  # noqa: E402
from train_chan_3 import preprocess as PP  # noqa: E402
from tabm import TabM  # noqa: E402
from rtdl_num_embeddings import LinearReLUEmbeddings  # noqa: E402


def log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with (OUT / "progress.txt").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def bss(pred: np.ndarray, y: np.ndarray) -> float:
    rate = float(np.mean(y))
    return float(100000.0 * (1.0 - np.mean((pred - y) ** 2) / (rate * (1.0 - rate))))


def opt_score(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    return FF.best_shift(np.asarray(pred, float), np.asarray(y, float))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def prep(x: np.ndarray, fit_mask: np.ndarray, cat_idx: list[int]):
    """The established F44 categorical / numeric preprocessing contract."""
    n, width = x.shape
    ci = np.asarray(cat_idx, dtype=np.int64)
    cset = set(ci.tolist())
    ni = np.asarray([j for j in range(width) if j not in cset], dtype=np.int64)
    xc = np.zeros((n, len(ci)), dtype=np.int64)
    cards: list[int] = []
    for k, j in enumerate(ci):
        vals = np.unique(x[fit_mask, j])
        vals = vals[~np.isnan(vals)]
        pos = np.clip(np.searchsorted(vals, x[:, j]), 0, max(len(vals) - 1, 0))
        hit = (vals[pos] == x[:, j]) if len(vals) else np.zeros(n, dtype=bool)
        xc[:, k] = np.where(hit, pos + 1, 0)
        cards.append(int(len(vals) + 1))
    xn = x[:, ni].astype(np.float64)
    missing = np.isnan(xn)
    median = np.nanmedian(xn[fit_mask], axis=0)
    xn = np.where(missing, median, xn)
    mean = xn[fit_mask].mean(axis=0)
    sd = xn[fit_mask].std(axis=0) + 1e-6
    xn = ((xn - mean) / sd).astype(np.float32)
    missing_cols = missing[fit_mask].any(axis=0)
    if missing_cols.any():
        xn = np.concatenate([xn, missing[:, missing_cols].astype(np.float32)], axis=1)
    return xn, xc, np.asarray(cards, dtype=np.int64)


def make_model(n_num: int, cards: np.ndarray, d_out: int) -> TabM:
    return TabM.make(
        n_num_features=n_num,
        cat_cardinalities=[int(x) for x in cards],
        d_out=d_out,
        num_embeddings=LinearReLUEmbeddings(n_num, d_embedding=16),
        arch_type="tabm", k=K, n_blocks=3, d_block=DBLOCK, dropout=0.1,
    ).to(DEVICE)


def make_paired_adapter(n_num: int, cards: np.ndarray, seed: int) -> TabM:
    """Give adapter head zero the exact d_out=1 initialization.

    `TabM.make(..., d_out=5)` consumes random numbers for four extra output
    columns before it initializes some BatchEnsemble factors.  Without this
    copy, a one-seed comparison can accidentally compare two different random
    models rather than `anchor` versus `anchor + adapters`.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    reference = make_model(n_num, cards, 1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    adapter = make_model(n_num, cards, 5)
    source, target = reference.state_dict(), adapter.state_dict()
    with torch.no_grad():
        for key, old in source.items():
            if old.shape == target[key].shape:
                target[key].copy_(old)
            elif key.startswith("output.") and old.ndim == target[key].ndim:
                # Different tabm versions lay out EnsembleLinear weights as
                # either (in, out) or (out, in).  Locate the unique output
                # dimension instead of assuming it is the final axis.
                changed = [j for j, (a, b) in enumerate(zip(old.shape, target[key].shape))
                           if a != b]
                if (len(changed) == 1 and old.shape[changed[0]] == 1
                        and target[key].shape[changed[0]] == 5
                        and all(old.shape[j] == target[key].shape[j]
                                for j in range(old.ndim) if j != changed[0])):
                    take = [slice(None)] * old.ndim
                    take[changed[0]] = slice(0, 1)
                    target[key][tuple(take)].copy_(old)
    adapter.load_state_dict(target)
    del reference
    torch.cuda.empty_cache()
    return adapter


def per_row_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean member BCE/Brier loss, retaining a row and head dimension."""
    target = target[:, None, None].expand_as(logits)
    bce = nnf.binary_cross_entropy_with_logits(logits, target, reduction="none")
    brier = (logits.sigmoid() - target).square()
    return (0.5 * bce + 0.5 * brier).mean(dim=1)  # (batch, heads)


def train_model(
    model: TabM,
    idx: np.ndarray,
    xn: torch.Tensor,
    xc: torch.Tensor,
    y: np.ndarray,
    gates: np.ndarray | None,
    sample_weight: np.ndarray | None,
    *,
    epochs: int,
    lr: float,
    seed: int,
    stage2: bool = False,
    tag: str = "",
) -> None:
    """Train anchor and, when supplied, four gate-weighted adapter heads."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if stage2:
        for p in model.parameters():
            p.requires_grad_(False)
        params = list(model.output.parameters()) + list(model.backbone.blocks[-1][0].parameters())
        for p in params:
            p.requires_grad_(True)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    ii = torch.from_numpy(idx.astype(np.int64))
    yy = torch.from_numpy(y.astype(np.float32))
    gg = None if gates is None else torch.from_numpy(gates.astype(np.float32))
    ww = None if sample_weight is None else torch.from_numpy(sample_weight.astype(np.float32))
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(idx))
        total = 0.0
        for start in range(0, len(idx), BS):
            pos = order[start:start + BS]
            row = ii[pos]
            xb = xn[row].to(DEVICE, non_blocking=True)
            cb = xc[row].to(DEVICE, non_blocking=True)
            target = yy[row].to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
                logits = model(xb, cb)
                loss_by_head = per_row_loss(logits, target)
                base = loss_by_head[:, 0]
                if ww is not None:
                    wb = ww[pos].to(DEVICE, non_blocking=True)
                    loss = (base * wb).sum() / wb.sum().clamp_min(1e-6)
                else:
                    loss = base.mean()
                if gg is not None:
                    gb = gg[row].to(DEVICE, non_blocking=True)
                    weighted = []
                    for head in range(gb.shape[1]):
                        ge = gb[:, head]
                        weighted.append((loss_by_head[:, head + 1] * ge).sum() /
                                        ge.sum().clamp_min(1e-6))
                    loss = loss + ADAPTER_LOSS * torch.stack(weighted).mean()
                    # An adapter is a regime-local correction, not an
                    # independent replacement model.  Detaching the anchor
                    # keeps this term from moving the anchor head itself.
                    delta2 = (logits[:, :, 1:] - logits[:, :, :1].detach()).square().mean(dim=1)
                    anchored = []
                    for head in range(gb.shape[1]):
                        ge = gb[:, head]
                        anchored.append((delta2[:, head] * ge).sum() /
                                        ge.sum().clamp_min(1e-6))
                    loss = loss + ADAPTER_ANCHOR * torch.stack(anchored).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            total += float(loss.detach()) * len(row)
        scheduler.step()
        log(f"{tag} epoch={epoch + 1}/{epochs} loss={total / len(idx):.6f}")


@torch.no_grad()
def predict_probs(model: TabM, xn: torch.Tensor, xc: torch.Tensor, idx: np.ndarray) -> np.ndarray:
    model.eval()
    out = np.empty((len(idx), model.output.weight.shape[-1]), dtype=np.float64)
    ii = torch.from_numpy(idx.astype(np.int64))
    for start in range(0, len(idx), 8192):
        row = ii[start:start + 8192]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP):
            probs = model(xn[row].to(DEVICE), xc[row].to(DEVICE)).sigmoid().mean(dim=1)
        out[start:start + len(row)] = probs.float().cpu().numpy()
    return out


def make_extra(raw: pd.DataFrame, x44: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Restored count evidence and the four legal, row-local adapter gates."""
    pn = raw["asof_pitcher_n"].to_numpy(np.float64)
    bn = raw["asof_batter_n"].to_numpy(np.float64)
    mn = raw["asof_pitcher_pitchmix_n"].to_numpy(np.float64)
    pn = np.nan_to_num(pn, nan=0.0)
    bn = np.nan_to_num(bn, nan=0.0)
    mn = np.nan_to_num(mn, nan=0.0)
    rp, rb = pn / (pn + 300.0), bn / (bn + 300.0)
    p_is = x44[:, names.index("p_is_succ")].astype(np.float64)
    b_is = x44[:, names.index("b_is_succ")].astype(np.float64)
    plat = x44[:, names.index("plat_dev")].astype(np.float64)
    all_extra = np.column_stack([
        np.log1p(pn), np.log1p(bn), np.log1p(mn),
        p_is * rp, b_is * rb, plat * rp,
    ]).astype(np.float32)
    if FEATURE_MODE == "count3":
        extra = all_extra[:, :3]
    elif FEATURE_MODE == "evidence3":
        extra = all_extra[:, 3:]
    elif FEATURE_MODE == "all6":
        extra = all_extra
    else:
        raise ValueError(f"unknown AV_FEATURE_MODE={FEATURE_MODE!r}")

    ph = raw["pitcher_hand"].to_numpy(np.float64)
    bh = raw["batter_hand"].to_numpy(np.float64)
    valid_hand = np.isin(ph, [1.0, 2.0]) & np.isin(bh, [1.0, 2.0])
    gates = np.column_stack([
        1.0 - rp,
        1.0 - rb,
        (valid_hand & (ph != bh)).astype(np.float32),
        (raw["strikes_before"].to_numpy(np.float64) == 2.0).astype(np.float32),
    ]).astype(np.float32)
    return extra, gates


def vote(anchor_and_adapters: np.ndarray, gates: np.ndarray, alpha: float) -> np.ndarray:
    """Anchor has unit weight; adapter weights are fixed, not fit on test rows."""
    anchor = anchor_and_adapters[:, 0]
    adapter = anchor_and_adapters[:, 1:]
    weights = alpha * gates
    return (anchor + (weights * adapter).sum(axis=1)) / (1.0 + weights.sum(axis=1))


def build_fold(vs: int):
    built = FF.build(str(DATA), VS=vs, return_frame=True)
    frame = built["frame"]
    season = built["season"].astype(np.int16)
    is_f = built["is_f"].astype(bool)
    y = built["y"].astype(np.float32)
    x44 = built["X44"].astype(np.float32)
    f44 = list(built["F44"])
    raw = pd.read_csv(
        DATA / "train.csv", encoding="utf-8-sig",
        usecols=["row_id", "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n",
                 "pitcher_hand", "batter_hand", "strikes_before"],
    )
    raw["_r"] = raw.row_id.astype(str).str.extract(r"(\d+)$", expand=False).astype(np.int64)
    raw = raw.sort_values("_r").reset_index(drop=True)
    assert np.array_equal(raw.row_id.to_numpy(), frame.row_id.to_numpy())

    # Same F44-compatible current preprocessing used for the deployed TabM paths.
    full = PP.sort_by_row_id(pd.read_csv(DATA / "train.csv", encoding="utf-8-sig"))
    hist = PP.fit_history_tables(full[full.season < vs])
    transformed = PP.transform_features(full, hist, train_mode=True)
    columns = list(transformed.columns)
    order = pd.Series(np.arange(len(full)), index=full.row_id.to_numpy()).reindex(raw.row_id).to_numpy()
    base = transformed.to_numpy(dtype=np.float32)[order]
    base[:, columns.index("plat_dev")] = x44[:, f44.index("plat_dev")]
    extra, gates = make_extra(raw, x44, f44)

    old = season <= OLD_F_MAX
    c4 = np.where(old & is_f, 0.0, np.where(old & ~is_f, 1.0,
                  np.where(is_f, 2.0, 3.0))).astype(np.float32)[:, None]
    cat_idx = [columns.index(c) for c in PP.TABM_CATEGORICAL_FEATURES] + [base.shape[1]]
    train = season < vs
    valid = season == vs
    return dict(
        vs=vs, y=y, season=season, is_f=is_f, train=train, valid=valid,
        base=np.concatenate([base, c4], axis=1),
        counts=np.concatenate([base, c4, extra], axis=1),
        cat_idx=cat_idx, gates=gates,
    )


def route_train(
    pack: dict,
    features: np.ndarray,
    use_adapters: bool,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Train the deployed all/regular/futures routes, then return valid logits."""
    train, valid = pack["train"], pack["valid"]
    xn_np, xc_np, cards = prep(features, train, pack["cat_idx"])
    xn, xc = torch.from_numpy(xn_np), torch.from_numpy(xc_np)
    tr_idx = np.flatnonzero(train)
    isf = pack["is_f"]
    old = pack["season"] <= OLD_F_MAX
    outputs: dict[str, np.ndarray] = {}
    n_out = 5 if use_adapters else 1
    for branch, selected, e2 in (
        ("all", np.ones(len(tr_idx), dtype=bool), 1),
        ("regular", ~isf[tr_idx], 1),
        ("futures", isf[tr_idx], 4),
    ):
        idx = tr_idx[selected]
        weight = None
        if branch in {"all", "futures"}:
            weight = np.where(isf[idx] & old[idx], OLD_F_WEIGHT, 1.0)
        # The comparison is paired at initialization as well as at minibatch
        # order.  Otherwise a favourable random draw can masquerade as an
        # adapter improvement.
        if use_adapters:
            model = make_paired_adapter(xn_np.shape[1], cards, seed)
        else:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = make_model(xn_np.shape[1], cards, n_out)
        branch_gates = pack["gates"] if use_adapters else None
        train_model(model, idx, xn, xc, pack["y"], branch_gates, weight,
                    epochs=2, lr=3e-3, seed=seed, tag=f"VS{pack['vs']} {branch} s{seed}")
        recent = idx[pack["season"][idx] == pack["vs"] - 1]
        for repeat in range(e2):
            train_model(model, recent, xn, xc, pack["y"], branch_gates, None,
                        epochs=1, lr=2e-4, seed=seed + repeat, stage2=True,
                        tag=f"VS{pack['vs']} {branch} s{seed} S2.{repeat + 1}")
        outputs[branch] = predict_probs(model, xn, xc, np.flatnonzero(valid))
        del model
        gc.collect()
        torch.cuda.empty_cache()

    v_isf = isf[valid]
    if use_adapters:
        vg = pack["gates"][valid]
        route_probs = np.where(
            v_isf[:, None],
            0.6 * outputs["all"] + 0.4 * outputs["futures"],
            0.6 * outputs["all"] + 0.4 * outputs["regular"],
        )
        route_logits = logit(route_probs)
        routed = {f"vote{a:.2f}": sigmoid(vote(route_logits, vg, a))
                  for a in (0.15, 0.30, 0.50)}
        routed["anchor"] = route_probs[:, 0]
    else:
        route_probs = np.where(
            v_isf,
            0.6 * outputs["all"][:, 0] + 0.4 * outputs["futures"][:, 0],
            0.6 * outputs["all"][:, 0] + 0.4 * outputs["regular"][:, 0],
        )
        routed = {"route": route_probs}
    return routed, outputs


def report(pack: dict, values: dict[str, list[np.ndarray]]) -> None:
    y = pack["y"][pack["valid"]].astype(float)
    isf = pack["is_f"][pack["valid"]]
    log(f"\n{'=' * 100}\nVS={pack['vs']}  n={len(y):,}  regular={(~isf).sum():,}  futures={isf.sum():,}")
    base = values["base44"]
    for name, pred_list in values.items():
        average = np.mean(pred_list, axis=0)
        score, shift = opt_score(average, y)
        reg, _ = opt_score(average[~isf], y[~isf])
        fut, _ = opt_score(average[isf], y[isf])
        raw = bss(average, y)
        line = (f"{name:14s} opt={score:8.2f} raw={raw:8.2f} shift={shift:+.5f} "
                f"R={reg:8.2f} F={fut:8.2f}")
        if name != "base44":
            diff = [opt_score(a, y)[0] - opt_score(b, y)[0] for a, b in zip(pred_list, base)]
            line += (f"  vs-base={np.mean(diff):+7.2f} "
                     f"{sum(x > 0 for x in diff)}/{len(diff)} "
                     f"[{', '.join(f'{x:+.1f}' for x in diff)}]")
        log(line)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this validation")
    log(f"device={torch.cuda.get_device_name()} folds={FOLDS} seeds={SEEDS} features={FEATURE_MODE} "
        f"k={K} d={DBLOCK} adapters={RUN_ADAPTERS} loss={ADAPTER_LOSS} anchor={ADAPTER_ANCHOR}")
    all_reports = {}
    for vs in FOLDS:
        pack = build_fold(vs)
        values: dict[str, list[np.ndarray]] = {"base44": [], "counts": []}
        if RUN_ADAPTERS:
            values.update({"anchor": [], "vote0.15": [], "vote0.30": [], "vote0.50": []})
        for seed in SEEDS:
            base, _ = route_train(pack, pack["base"], False, seed)
            counts, _ = route_train(pack, pack["counts"], False, seed)
            values["base44"].append(base["route"])
            values["counts"].append(counts["route"])
            if RUN_ADAPTERS:
                adapt, _ = route_train(pack, pack["counts"], True, seed)
                for name in ("anchor", "vote0.15", "vote0.30", "vote0.50"):
                    values[name].append(adapt[name])
                    np.save(OUT / f"av_{vs}_{name}_s{seed}.npy", adapt[name])
            np.save(OUT / f"av_{vs}_base44_s{seed}.npy", base["route"])
            np.save(OUT / f"av_{vs}_counts_s{seed}.npy", counts["route"])
        report(pack, values)
        all_reports[str(vs)] = {name: np.mean(pred, axis=0).tolist()[:0] for name, pred in values.items()}
    (OUT / "config.json").write_text(json.dumps({
        "folds": FOLDS, "seeds": SEEDS, "adapter_loss": ADAPTER_LOSS,
        "adapter_anchor": ADAPTER_ANCHOR,
        "run_adapters": RUN_ADAPTERS,
        "vote_alphas": [0.15, 0.30, 0.50],
        "count_features": ["log_pn", "log_bn", "log_pitchmix_n", "p_is_x_reliability",
                           "b_is_x_reliability", "plat_dev_x_reliability"],
        "gates": ["cold_pitcher", "cold_batter", "platoon", "two_strike"],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
