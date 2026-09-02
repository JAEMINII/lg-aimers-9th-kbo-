# -*- coding: utf-8 -*-
"""submit_41 의 조회표 결함이 실제로 점수를 깎는지 walk-forward 로 잰다.

결함
    script.py 의 add_pitcher_count12 / add_pitcher_count_hand 는
        pos = np.searchsorted(pair_keys, key)
        hit = (pos < len(pair_keys))          # 값이 같은지 안 본다
    라서 키가 표 중간에 없으면 **다음 투수의 칸**을 읽는다.
    학습 코드(train_c12_submit.add_cmh)는 keys[pos]==kv 로 제대로 거른다.
    즉 학습/추론 규약 불일치다 — 어제 1069->1076 을 만든 것과 같은 부류.

재는 법 — 제출을 쓰지 않는다
    학습   season < 2024 행. 피처는 strict as-of (자기 시즌 미만만 본다)
    추론   season == 2024 행. 피처는 **<2024 합산표**를 조회해서 만든다
           2025 가 2019~2024 표를 조회하는 것과 같은 관계다
    두 팔은 **같은 모델**로 예측한다. 다른 것은 조회 방식 하나뿐이다.
        buggy   pos < len            (지금 실려 있는 것)
        fixed   pos < len & keys[pos]==key

주의
    최적 시프트에서 비교한다. 고정 시프트로 재면 수준 보정과 판별력이 섞인다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "colab"))
sys.path.insert(0, os.path.join(ROOT, "nn_experiments"))
os.environ.setdefault("AIMERS_ROOT", ROOT)
DATA = os.environ.get("AIMERS_DATA", os.path.join(ROOT, "open (1)", "data"))
VS = int(os.environ.get("VS", "2024"))
SEEDS = (1, 42, 777)
ALPHA = 200.0
DECAY = 2.0

import features44 as F                                          # noqa: E402
import train_c12_submit as T                                    # noqa: E402
from catboost import CatBoostClassifier                         # noqa: E402


def build_tables(d, keycol, mul, tag):
    """그 부분집합만으로 합산표를 만든다 (추론이 조회할 표)."""
    tot = d.groupby(["pitcher_id", keycol], sort=False)["control_success"] \
           .agg(["size", "sum"]).sort_index()
    pt = d.groupby("pitcher_id", sort=False)["control_success"] \
          .agg(["size", "sum"]).sort_index()
    keys = (tot.index.get_level_values(0).to_numpy(np.int64) * mul
            + tot.index.get_level_values(1).to_numpy(np.int64))
    g = float(d["control_success"].mean())
    pr = d.groupby(keycol)["control_success"].mean()
    return {f"{tag}_key": keys, f"{tag}_n": tot["size"].to_numpy(np.float64),
            f"{tag}_s": tot["sum"].to_numpy(np.float64),
            f"{tag}_pid": pt.index.to_numpy(np.int64),
            f"{tag}_tn": pt["size"].to_numpy(np.float64),
            f"{tag}_ts": pt["sum"].to_numpy(np.float64),
            f"{tag}_prior_keys": pr.index.to_numpy(np.int64),
            f"{tag}_prior": pr.to_numpy(np.float64), f"{tag}_g": g}


def lookup(tb, tag, pid, cell, mul, strict):
    """script.py 와 같은 조회. strict=False 가 지금 실려 있는 판이다."""
    key = pid * mul + cell
    ks = tb[f"{tag}_key"]
    pos = np.searchsorted(ks, key)
    safe = np.minimum(pos, len(ks) - 1)
    hit = pos < len(ks)
    if strict:
        hit = hit & (ks[safe] == key)
    n = np.where(hit, tb[f"{tag}_n"][safe], 0.0)
    s = np.where(hit, tb[f"{tag}_s"][safe], 0.0)
    pids = tb[f"{tag}_pid"]
    pp = np.searchsorted(pids, pid)
    psafe = np.minimum(pp, len(pids) - 1)
    ph = pp < len(pids)
    if strict:
        ph = ph & (pids[psafe] == pid)
    n_all = np.where(ph, tb[f"{tag}_tn"][psafe], 0.0)
    s_all = np.where(ph, tb[f"{tag}_ts"][psafe], 0.0)
    g = tb[f"{tag}_g"]
    pk, pv = tb[f"{tag}_prior_keys"], tb[f"{tag}_prior"]
    q = np.searchsorted(pk, cell)
    qs = np.minimum(q, len(pk) - 1)
    prior = np.where((q < len(pk)) & (pk[qs] == cell), pv[qs], g)
    base = (s_all + ALPHA * g) / (n_all + ALPHA)
    rate = (s + ALPHA * prior) / (n + ALPHA)
    return rate, rate - base, np.log1p(n)


if __name__ == "__main__":
    t0 = time.time()
    built = F.build(DATA, VS=VS, return_frame=True)
    fr = built["frame"]
    base_f = list(built["F44"])
    fr, _ = T.add_c12(fr, return_tables=True)
    fr, _ = T.add_cmh(fr, return_tables=True)
    feats = base_f + ["pc_c12_rate", "pc_c12_dev", "pc_c12_n",
                      "pc_cmh_rate", "pc_cmh_dev", "pc_cmh_n"]
    season = fr["season"].to_numpy()
    y = fr["control_success"].to_numpy(np.float64)
    gt = fr["game_type"].to_numpy()          # features44 가 R=0 / F=1 로 고쳐준다
    tr = season < VS
    te = season == VS
    print(f"  자료 {time.time()-t0:.0f}초   학습 {int(tr.sum()):,}  검증 {int(te.sum()):,}")

    hist = fr[tr]
    tb = {}
    tb.update(build_tables(hist, "cnt12", 16, "pc"))
    tb.update(build_tables(hist, "pcmh", 32, "ph"))
    pid = fr["pitcher_id"].to_numpy(np.int64)[te]
    c12 = fr["cnt12"].to_numpy(np.int64)[te]
    cmh = fr["pcmh"].to_numpy(np.int64)[te]

    X = fr[feats].to_numpy(np.float32)
    arms = {}
    for nm, strict in (("buggy(현행)", False), ("fixed", True)):
        Xa = X[te].copy()
        for tag, cell, mul, cols in (("pc", c12, 16, (44, 45, 46)),
                                     ("ph", cmh, 32, (47, 48, 49))):
            r, dv, ln = lookup(tb, tag, pid, cell, mul, strict)
            for j, v in zip(cols, (r, dv, ln)):
                Xa[:, j] = v.astype(np.float32)
        arms[nm] = Xa
    diff = np.abs(arms["buggy(현행)"] - arms["fixed"]).max(1)
    nd = int((diff > 0).sum())
    print(f"  두 팔이 다른 행 {nd:,} / {int(te.sum()):,} ({nd/te.sum()*100:.2f}%)  "
          f"최대차 {diff.max():.4f}")

    w = DECAY ** (fr["season"].to_numpy(np.float64)[tr] - 2019.0)
    hp = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
              verbose=0, allow_writing_files=False, task_type="GPU", devices="0")
    masks = {"all": np.ones(int(tr.sum()), bool),
             "regular": gt[tr] == 0, "futures": gt[tr] == 1}
    P = {k: {g: [] for g in masks} for k in arms}
    for g, m in masks.items():
        for sd in SEEDS:
            t1 = time.time()
            mo = CatBoostClassifier(random_seed=sd, **hp)
            mo.fit(X[tr][m], y[tr][m], sample_weight=w[m])
            for k, Xa in arms.items():
                P[k][g].append(mo.predict_proba(Xa)[:, 1])
            print(f"    {g:8s} seed {sd:>3d}  {time.time()-t1:5.0f}초", flush=True)

    isf = gt[te] == 1
    yv = y[te]
    print(f"\n{'='*76}\n  조회표 결함 수정 — walk-forward VS={VS}, 같은 모델·조회만 다름"
          f"\n{'='*76}")
    print(f"  {'팔':12s} {'전체':>9s} {'1군':>9s} {'퓨처스':>9s}")
    sc = lambda p, m: F.best_shift(p[m], yv[m])[0]
    allm = np.ones(len(yv), bool)
    per_seed = {}
    for k in arms:
        ps = [np.where(isf, 0.6 * P[k]["all"][i] + 0.4 * P[k]["futures"][i],
                       0.6 * P[k]["all"][i] + 0.4 * P[k]["regular"][i])
              for i in range(len(SEEDS))]
        per_seed[k] = ps
        p = np.mean(ps, 0)
        print(f"  {k:12s} {sc(p,allm):9.1f} {sc(p,~isf):9.1f} {sc(p,isf):9.1f}")
    dd = [sc(a, allm) - sc(b, allm)
          for a, b in zip(per_seed["fixed"], per_seed["buggy(현행)"])]
    mu = float(np.mean(dd)); se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
    print(f"\n  fixed - buggy   {mu:+6.1f} +- {se:4.1f}  t={mu/max(se,1e-9):5.2f}  "
          f"{sum(1 for v in dd if v>0)}/{len(dd)}   [{' '.join(f'{v:+.1f}' for v in dd)}]")
    p_bug = np.mean(per_seed["buggy(현행)"], axis=0)
    p_fix = np.mean(per_seed["fixed"], axis=0)
    save_pred = os.environ.get("SAVE_PRED")
    if save_pred:
        np.save(save_pred, p_fix.astype(np.float32))
        print(f"  saved fixed route predictions: {save_pred}")
    _, shared_shift = F.best_shift(p_bug, yv)
    raw_delta = F.bss(p_fix, yv) - F.bss(p_bug, yv)
    shared_delta = (F.bss(F.shift(p_fix, shared_shift), yv)
                    - F.bss(F.shift(p_bug, shared_shift), yv))
    print(f"  raw Brier delta  {raw_delta:+6.2f}   "
          f"buggy shift={shared_shift:+.4f} shared-shift delta={shared_delta:+6.2f}")
    print("  raw/shared-shift는 재보정을 허용하지 않은 비교다. 조회 규약 수정의 방향만 판단한다.")
