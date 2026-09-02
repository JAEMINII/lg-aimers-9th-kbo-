# -*- coding: utf-8 -*-
"""트랙맨 피처를 믿을 수 있는 관문에서 잰다.

시간 규약
    season Y 의 행에는 '시즌 < Y' 의 트랙맨만 쓴다. 누적 평균이다.
    2025 제출 시엔 2019~2024 전부가 쓰이므로 배치와 구조가 같다.
    트랙맨엔 정답이 없어 자기정답 누출은 불가능하고, 시간 누출만 이렇게 막는다.

결측 처리
    매핑 안 된 투수(train 행의 6.3%)와 데뷔 시즌 투수는 값이 없다.
    NaN 으로 두면 G.prep 이 중앙값으로 채우고 결측 플래그를 붙인다.
    '없다' 는 것도 정보다 — 트랙맨에 안 잡히는 투수는 등판이 적은 투수다.

세 벌을 비교한다
    base    44열
    tm      44 + 트랙맨 13
    tm_2s   44 + 2스트라이크 전용 8개
    tm_sd2  44 + 릴리스 흔들림 2개만 (tm_relh_sd, tm_rels_sd)
            제구 과제에 가장 직접적인 가설. 13개가 노이즈를 부르면 이쪽이 산다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
SEEDS = (42, 1, 777)

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)
VS = int(os.environ.get("VS", 2024))


def trackman_matrix(tr_seasons, tr_pitchers, cols=None):
    """행별 트랙맨 피처. season Y 행에는 시즌 < Y 누적 평균만 넣는다."""
    t = pd.read_csv(os.path.join(SC, "trackman_pitcher_season.csv"))
    feat = [c for c in t.columns if c.startswith("tm_")]
    if cols is not None:
        feat = [c for c in feat if c in cols]
    # tm_n 가중 누적: 시즌별 평균을 투구수로 가중해 '시즌 < Y' 까지 합친다
    t = t.sort_values(["pitcher_id", "season"])
    w = t["tm_n"].to_numpy(dtype=np.float64)
    acc = {}
    for c in feat:
        v = t[c].to_numpy(dtype=np.float64)
        ok = ~np.isnan(v)
        num = pd.Series(np.where(ok, v * w, 0.0)).groupby(
            [t.pitcher_id.values, t.season.values]).sum()
        den = pd.Series(np.where(ok, w, 0.0)).groupby(
            [t.pitcher_id.values, t.season.values]).sum()
        acc[c] = (num, den)
    # (투수, 시즌) -> 그 시즌까지 포함한 누적. 뒤에서 한 시즌 밀어 '이전까지' 로 만든다
    seasons = sorted(t.season.unique())
    rows = {}
    for pid in t.pitcher_id.unique():
        cum_n = {c: 0.0 for c in feat}
        cum_d = {c: 0.0 for c in feat}
        for s in seasons:
            rows[(pid, s)] = {c: (cum_n[c] / cum_d[c] if cum_d[c] > 0 else np.nan)
                              for c in feat}
            for c in feat:
                key = (pid, s)
                if key in acc[c][0].index:
                    cum_n[c] += float(acc[c][0].loc[key])
                    cum_d[c] += float(acc[c][1].loc[key])
    M = np.full((len(tr_seasons), len(feat)), np.nan, dtype=np.float32)
    for i, (pid, s) in enumerate(zip(tr_pitchers, tr_seasons)):
        r = rows.get((pid, s))
        if r is not None:
            M[i] = [r[c] for c in feat]
    return M, feat


def one(X, cat_idx, seed):
    Xn, Xc, cards = G.prep(X, G.m_tr, cat_idx)
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    P = {}
    tr_idx = np.where(G.m_tr)[0]
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("futures", G.is_f[tr_idx]), ("regular", ~G.is_f[tr_idx])):
        idx = tr_idx[sel]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model()
        G.train(m, idx, 2, 2e-3, seed=seed, tag=f"{br} S1")
        s2 = G.season[idx] == VS - 1
        G.train(m, idx[s2], 1, 2e-4, params=G.stage2_params(m), seed=seed,
                tag=f"{br} S2")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    return np.where(is_f, 0.4 * P["futures"] + 0.6 * P["all"],
                    0.4 * P["regular"] + 0.6 * P["all"])


if __name__ == "__main__":
    d = F.build(DATA, VS=VS)
    base = d["X44"]
    raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                      usecols=["pitcher_id", "season"])
    M_all, names_all = trackman_matrix(raw.season.to_numpy(),
                                       raw.pitcher_id.to_numpy())
    M_min, names_min = trackman_matrix(raw.season.to_numpy(),
                                       raw.pitcher_id.to_numpy(),
                                       cols={"tm_relh_sd", "tm_rels_sd"})
    G.log(f"\n  트랙맨 {len(names_all)}개  {names_all}")
    G.log(f"  값이 있는 행 비율 {(~np.isnan(M_all[:,0])).mean()*100:.1f}%  "
          f"(관문 {(~np.isnan(M_all[gate,0])).mean()*100:.1f}%)")

    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    MLPF = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    G.log("\n" + "=" * 70)
    G.log("  구성       열    시드평균 단독   제출비중   개별 시드")
    G.log("=" * 70)
    ref = None
    # 2스트라이크 전용만 따로. 오차의 28.9%가 그 국면이고 TabM 이 유독 약하다.
    two = {c for c in names_all if c.startswith("tm2s")}
    M_2s, _ = trackman_matrix(raw.season.to_numpy(), raw.pitcher_id.to_numpy(),
                              cols=two)
    for name, X in (("base", base),
                    ("tm_all", np.concatenate([base, M_all], 1)),
                    ("tm_sd2", np.concatenate([base, M_min], 1)),
                    ("tm_2s", np.concatenate([base, M_2s], 1))):
        t0 = time.time()
        ps = [one(X, d["cat_idx"], sd) for sd in SEEDS]
        solos = [F.best_shift(p[FULL], yv[FULL])[0] for p in ps]
        r = np.mean(ps, 0)
        solo = F.best_shift(r[FULL], yv[FULL])[0]
        bl = F.best_shift((0.10 * CB + 0.30 * MLPF + 0.60 * r)[FULL], yv[FULL])[0]
        np.save(os.path.join(OUT, f"tm_{name}.npy"), r)
        dl = "" if ref is None else f"   기준대비 {solo-ref[0]:+6.1f} / {bl-ref[1]:+6.1f}"
        if ref is None:
            ref = (solo, bl)
        G.log(f"  {name:8s} {X.shape[1]:4d}   {solo:7.1f}      {bl:7.1f}   "
              f"[{', '.join(f'{v:.0f}' for v in solos)}]   {time.time()-t0:.0f}s{dl}")
    G.log("\n  +5 미만은 판정 보류.")
