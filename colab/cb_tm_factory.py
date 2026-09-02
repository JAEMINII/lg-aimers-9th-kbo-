# -*- coding: utf-8 -*-
"""트랙맨 세쌍 공장 — (투수, 카운트12) 조건부 구종 성향을 CatBoost 관문에 올린다.

왜 이 칸인가
    증명된 축     투수x카운트12 (pc_c12 가 LB +7)
    증명된 형식   (rate, dev, n) 세쌍 + CatBoost
    새 라벨원     트랙맨 구종 (fastball/breaking/offspeed) — 결과 라벨이 아니라
                 **행동 라벨**이다. "3-2 에서 이 투수는 변화구를 던지는 투수인가"
    기존 44열의 asof_pitcher_fastball_rate 는 카운트 무관 전체 비율이라 이게 없다.
    행 매칭 불필요 — 트랙맨 자체의 카운트 열 + 투수 매핑(443명)만 쓴다.

합법성
    트랙맨은 제공 데이터. 표는 시즌 cumsum-shift 로 as-of (관문은 <VS, 배치는
    <=2024 전부 = 2025 의 과거). 조회는 그 행의 (투수, 카운트) 뿐 — 규칙 4 무관.

팔 (하나씩, 묶지 않는다)
    base       배치 50열
    tmix_fb    + P(fastball | 투수, 카운트) 세쌍
    tmix_br    + P(breaking | ...) 세쌍
    tmix_off   + P(offspeed | ...) 세쌍
    relsd      + 투수 릴리스 일관성 (rel_height/rel_side 표준편차, as-of) 3열
판정  VS=2024 스크린 -> 통과분만 VS=2022. 배치 혼합 효과까지 낸다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, SC)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'nn_experiments'))
os.environ.setdefault("AIMERS_ROOT", ROOT)
DATA = os.path.join(ROOT, "open (1)", "data")
MAPDIR = os.path.join(ROOT, "trackman_map")
DL = os.path.join(SC, "_dl")
VS = int(os.environ.get("TF_VS", "2024"))
SEEDS = (1, 42, 777)
ALPHA, DECAY = 100.0, 2.0

import features44 as F                                          # noqa: E402
import train_c12_submit as T                                    # noqa: E402
from catboost import CatBoostClassifier                         # noqa: E402


def build_tm_tables():
    """트랙맨에서 (투수id, 시즌, 카운트12) 구종/릴리스 집계. 시즌 as-of 는 호출부."""
    pm = pd.read_csv(os.path.join(MAPDIR, "pitcher_map.csv"), encoding="utf-8-sig")
    t2p = dict(zip(pm["trackman_id"], pm["pitcher_id"]))
    tm = pd.read_csv(os.path.join(DATA, "trackman_history.csv"),
                     encoding="utf-8-sig",
                     usecols=["season", "pitcher_trackman_id", "balls_before",
                              "strikes_before", "pitch_type_group",
                              "rel_height", "rel_side"])
    tm["pid"] = tm["pitcher_trackman_id"].map(t2p)
    tm = tm.dropna(subset=["pid"]).copy()
    tm["pid"] = tm["pid"].astype(int)
    tm["c12"] = np.clip(tm["balls_before"].astype(int) * 3
                        + tm["strikes_before"].astype(int), 0, 11)
    for g in ("fastball", "breaking", "offspeed"):
        tm[f"is_{g}"] = (tm["pitch_type_group"] == g).astype(float)
    print(f"  트랙맨 매핑행 {len(tm):,}  투수 {tm.pid.nunique()}")
    return tm


def triple_from_tm(tm, fr, label, vs):
    """P(label | 투수, c12) - P(label | 투수), as-of(<vs 시즌 누적)."""
    d = tm[tm["season"] < vs]
    cell = d.groupby(["pid", "c12"])[f"is_{label}"].agg(["size", "sum"])
    tot = d.groupby("pid")[f"is_{label}"].agg(["size", "sum"])
    g = float(d[f"is_{label}"].mean())
    prior = d.groupby("c12")[f"is_{label}"].mean()
    pid = fr["pitcher_id"].to_numpy()
    c12 = fr["cnt12"].to_numpy(np.int64)
    key = pd.MultiIndex.from_arrays([pid, c12])
    cs = cell.reindex(key)
    n = np.nan_to_num(cs["size"].to_numpy())
    s = np.nan_to_num(cs["sum"].to_numpy())
    ts = tot.reindex(pid)
    n_all = np.nan_to_num(ts["size"].to_numpy())
    s_all = np.nan_to_num(ts["sum"].to_numpy())
    pr = prior.reindex(c12).fillna(g).to_numpy()
    rate = (s + ALPHA * pr) / (n + ALPHA)
    base = (s_all + ALPHA * g) / (n_all + ALPHA)
    return {f"tmx_{label}_rate": rate.astype("float32"),
            f"tmx_{label}_dev": (rate - base).astype("float32"),
            f"tmx_{label}_n": np.log1p(n).astype("float32")}


def relsd_from_tm(tm, fr, vs):
    d = tm[tm["season"] < vs]
    a = d.groupby("pid").agg(hsd=("rel_height", "std"), ssd=("rel_side", "std"),
                             n=("rel_height", "size"))
    pid = fr["pitcher_id"].to_numpy()
    r = a.reindex(pid)
    return {"tmr_hsd": r["hsd"].to_numpy(np.float32),
            "tmr_ssd": r["ssd"].to_numpy(np.float32),
            "tmr_n": np.log1p(np.nan_to_num(r["n"].to_numpy())).astype("float32")}


if __name__ == "__main__":
    built = F.build(DATA, VS=VS, return_frame=True)
    fr = built["frame"]
    base_f = list(built["F44"])
    fr, _ = T.add_c12(fr, return_tables=True)
    fr, _ = T.add_cmh(fr, return_tables=True)
    BASE50 = base_f + ["pc_c12_rate", "pc_c12_dev", "pc_c12_n",
                       "pc_cmh_rate", "pc_cmh_dev", "pc_cmh_n"]
    tm = build_tm_tables()
    CAND = {}
    for lab in ("fastball", "breaking", "offspeed"):
        cols = triple_from_tm(tm, fr, lab, VS)
        for c, v in cols.items():
            fr[c] = v
        CAND[f"tmix_{lab[:2]}"] = list(cols)
    rc = relsd_from_tm(tm, fr, VS)
    for c, v in rc.items():
        fr[c] = v
    CAND["relsd"] = list(rc)
    for nm, cs in CAND.items():
        dv = fr[cs[1]] if len(cs) > 1 else fr[cs[0]]
        print(f"  {nm:10s} {cs}  sd {np.nanstd(fr[cs[0]].to_numpy()):.4f}")

    y = fr["control_success"].to_numpy(np.float64)
    season = fr["season"].to_numpy()
    gt = fr["game_type"].to_numpy()
    tr_m, te_m = season < VS, season == VS
    w = DECAY ** (season[tr_m].astype(np.float64) - 2019.0)
    isf_t = gt[te_m] == 1
    yv = y[te_m]
    hp = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
              verbose=0, allow_writing_files=False)
    masks = {"all": np.ones(int(tr_m.sum()), bool),
             "regular": gt[tr_m] == 0, "futures": gt[tr_m] == 1}

    def sc(p, m=None):
        return F.best_shift(p if m is None else p[m],
                            yv if m is None else yv[m])[0]

    tab = np.load(DL + "/c4g_2024_c4_all_s42.npy") * 0.6 \
        + np.load(DL + "/c4g_2024_c4_regular_s42.npy") * 0.4 \
        if VS == 2024 else np.load(DL + "/h2h_2022_friend_s42.npy")
    din = np.load(DL + f"/dg_{VS}_DIN.npy").mean(0)
    ms = np.load(DL + f"/msaudit/audit_all_vs{VS}_direct.npy").astype(np.float64)

    ARMS = [("base", [])] + [(nm, cs) for nm, cs in CAND.items()]
    R = {}
    for nm, extra in ARMS:
        feats = BASE50 + extra
        X = fr[feats].to_numpy(np.float32)
        t0 = time.time()
        pg = {g: [] for g in masks}
        for g, m in masks.items():
            for sd in SEEDS:
                mo = CatBoostClassifier(random_seed=sd, **hp)
                mo.fit(X[tr_m][m], y[tr_m][m], sample_weight=w[m])
                pg[g].append(mo.predict_proba(X[te_m])[:, 1])
        cbP = [np.where(isf_t, 0.6 * pg["all"][i] + 0.4 * pg["futures"][i],
                        0.6 * pg["all"][i] + 0.4 * pg["regular"][i])
               for i in range(len(SEEDS))]
        R[nm] = cbP
        np.save(DL + f"/tf_{VS}_{nm}.npy", np.asarray(cbP))
        print(f"    {nm:10s} 학습 {time.time() - t0:5.0f}s", flush=True)

    ref = R["base"]
    b48 = [F.best_shift(0.31 * np.mean(ref, 0) + 0.13 * tab + 0.31 * din
                        + 0.25 * ms, yv)[0]]
    print("\n" + "=" * 92)
    print(f"  트랙맨 세쌍 공장 VS={VS}   기준 = 배치50열   혼합 = 48식 4원")
    print("=" * 92)
    for nm, _ in ARMS:
        p = np.mean(R[nm], 0)
        dd = [sc(a) - sc(b) for a, b in zip(R[nm], ref)]
        mix = F.best_shift(0.31 * p + 0.13 * tab + 0.31 * din + 0.25 * ms,
                           yv)[0] - b48[0]
        print(f"  {nm:10s} CB단독 {sc(p):8.1f}   차이 "
              + " ".join(f"{v:+6.1f}" for v in dd)
              + f"  {sum(1 for v in dd if v > 0)}/3   4원혼합 {mix:+6.1f}",
              flush=True)
    print("\n  통과분만 VS=2022 로 확정한다.")
