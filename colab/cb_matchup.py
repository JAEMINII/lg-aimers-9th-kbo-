# -*- coding: utf-8 -*-
"""CatBoost 세쌍 공장 — 증명된 제조법을 안 해본 칸에 일괄 적용한다.

제조법 (pc_c12/pc_cmh 로 리더보드 +7 실증)
    (rate, dev, n) 세 쌍, 시즌 cumsum->shift(1) 로 as-of, ALPHA=200,
    CatBoost 에 추가 (id 임베딩이 없어 세 쌍이 새 정보다).

칸 선정 근거
    축은 증명된 것만: 투수x카운트12 (+7), 타자손 (plat_dev +11).
    라벨은 복원 5종: success/ball/strike/reverse/middle (99.94% 복원).
    편차 한 열 + 편상관으로 기각된 조합이라도 세 쌍 + CatBoost 에선 다르다.
    success x cnt 가 편상관 -0.00114 인데 세 쌍으로 +7 이었다.

기준선   배치 50열 그대로 (44 + pc_c12 + pc_cmh). 후보는 그 위에 하나씩.
채점     CatBoost 라우팅 단독 + 전체 혼합(현행 가중, 재조정 가중 둘 다).
판정     VS=2024 스크린 -> 통과분만 VS=2022 확정.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
for p in (SC, "/root/aimers", os.path.dirname(SC)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("AIMERS_ROOT", "/root")
DATA = os.environ.get("AIMERS_DATA", "/root/open (1)/data")
DL = os.environ.get("AIMERS_DL", "/root/aimers/_dl")
VS = int(os.environ.get("CF_VS", "2024"))
SEEDS = (1, 42, 777)
ALPHA, DECAY = 200.0, 2.0

import features44 as F                                          # noqa: E402
import train_c12_submit as T                                    # noqa: E402
from catboost import CatBoostClassifier                         # noqa: E402


def recover(fr, col):
    """as-of 누적비율에서 투구 단위 라벨을 되찾는다. nan = 복원 불가."""
    n = fr["asof_pitcher_n"].to_numpy(np.float64)
    pid = fr["pitcher_id"].to_numpy()
    ok = np.r_[False, (pid[1:] == pid[:-1]) & (np.diff(n) == 1)]
    src = np.where(ok)[0]
    dst = src - 1
    cum = fr[col].to_numpy(np.float64) * n
    inc = cum[src] - cum[dst]
    lab = np.round(inc)
    good = (np.abs(inc - lab) < 0.25) & ((lab == 0) | (lab == 1))
    v = np.full(len(fr), np.nan)
    v[dst[good]] = lab[good]
    return v


def triple(fr, ent, keyvals, v, tag):
    """(rate, dev, n) 세 쌍. v 는 nan 마스크 있는 라벨."""
    have = np.isfinite(v)
    z = pd.DataFrame({"e": fr[ent].to_numpy(), "season": fr["season"].to_numpy(),
                      "k": np.asarray(keyvals),
                      "v": np.where(have, v, 0.0), "n": have.astype(float)})
    b = z.groupby(["e", "season", "k"], sort=False)[["v", "n"]].sum() \
         .unstack("k", fill_value=0.0)
    b = b.reindex(sorted(b.columns), axis=1)
    keys = np.array(sorted({k for _, k in b.columns}), np.int64)
    cc = b.groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0.0)
    idx = pd.MultiIndex.from_arrays([z["e"].to_numpy(), z["season"].to_numpy()])
    sv = cc["v"].reindex(idx).fillna(0.0).to_numpy()
    sn = cc["n"].reindex(idx).fillna(0.0).to_numpy()
    pos = np.clip(np.searchsorted(keys, np.asarray(keyvals)), 0, len(keys) - 1)
    hit = keys[pos] == np.asarray(keyvals)
    rr = np.arange(len(z))
    s = np.where(hit, sv[rr, pos], 0.0)
    n = np.where(hit, sn[rr, pos], 0.0)
    g = float(np.nansum(v)) / max(float(have.sum()), 1.0)
    pr = z.groupby("k").apply(lambda t: t["v"].sum() / max(t["n"].sum(), 1.0),
                              include_groups=False)
    pr = pr.reindex(keys).fillna(g).to_numpy()[pos]
    n_all, s_all = sn.sum(1), sv.sum(1)
    base = (s_all + ALPHA * g) / (n_all + ALPHA)
    rate = (s + ALPHA * pr) / (n + ALPHA)
    return {tag + "_rate": rate.astype("float32"),
            tag + "_dev": (rate - base).astype("float32"),
            tag + "_n": np.log1p(n).astype("float32")}


if __name__ == "__main__":
    built = F.build(DATA, VS=VS, return_frame=True)
    fr = built["frame"]
    base_f = list(built["F44"])
    fr, _ = T.add_c12(fr, return_tables=True)
    fr, _ = T.add_cmh(fr, return_tables=True)
    BASE50 = base_f + ["pc_c12_rate", "pc_c12_dev", "pc_c12_n",
                       "pc_cmh_rate", "pc_cmh_dev", "pc_cmh_n"]
    y = fr["control_success"].to_numpy(np.float64)
    season = fr["season"].to_numpy()
    gt = fr["game_type"].to_numpy()
    cnt = fr["cnt12"].to_numpy(np.int64)
    bh = fr["batter_hand"].to_numpy(np.int64)
    ph = fr["pitcher_hand"].to_numpy(np.int64)
    lev = np.minimum(np.abs(fr["score_diff_pitcher_team"].fillna(0)
                            .to_numpy(np.int64)), 4)
    LAB = {nm: recover(fr, c) for nm, c in
           (("ball", "asof_pitcher_ball_rate"),
            ("strike", "asof_pitcher_strike_rate"),
            ("rev", "asof_pitcher_reverse_rate"))}
    succ = y.astype(np.float64)          # 라벨로도 쓴다 (항상 안다)

    # 매치업 판: 축 키를 batter_id 로 (투수 개체 x 타자 개체 — 마지막 미실험 셀)
    bid = fr["batter_id"].to_numpy(np.int64)
    CAND = {
        "pc_bt": ("pitcher_id", bid, succ, "성공 x 개별타자 (매치업)"),
    }
    for tag, (ent, kv, v, desc) in CAND.items():
        for c, arr in triple(fr, ent, kv, v, tag).items():
            fr[c] = arr
        print("  %-14s %-28s dev sd %.5f"
              % (tag, desc, fr[tag + "_dev"].std()), flush=True)

    tr_m, te_m = season < VS, season == VS
    w = DECAY ** (season[tr_m].astype(np.float64) - 2019.0)
    isf_t = gt[te_m] == 1
    yv = y[te_m]
    hp = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
              verbose=0, allow_writing_files=False, task_type="GPU", devices="0")
    masks = {"all": np.ones(int(tr_m.sum()), bool),
             "regular": gt[tr_m] == 0, "futures": gt[tr_m] == 1}

    def sc(p, m=None):
        return F.best_shift(p if m is None else p[m],
                            yv if m is None else yv[m])[0]

    tab = np.load(DL + "/h2h_%d_friend_s42.npy" % VS).astype(np.float64)
    din = np.load(DL + "/dg_%d_DIN.npy" % VS).mean(0)

    ARMS = [("base", [])] + [(t, [t + "_rate", t + "_dev", t + "_n"])
                             for t in CAND]
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
        np.save(DL + "/cfm_%d_%s.npy" % (VS, nm), np.asarray(cbP))
        print("    %-14s 학습 %5.0fs" % (nm, time.time() - t0), flush=True)

    ref = R["base"]
    print("\n" + "=" * 96)
    print("  세쌍 공장 VS=%d   기준선 = 배치 50열   혼합: 현행(.225/.525/.25) / "
          "재조정(.40/.20/.40)" % VS)
    print("=" * 96)
    print("  %-14s %8s   %-24s %4s   %9s %10s"
          % ("팔", "CB단독", "CB차이 (시드별)", "부호", "혼합(현행)", "혼합(재조정)"))
    b1 = sc(0.225 * np.mean(ref, 0) + 0.525 * tab + 0.25 * din)
    b2 = sc(0.40 * np.mean(ref, 0) + 0.20 * tab + 0.40 * din)
    for nm, _ in ARMS:
        p = np.mean(R[nm], 0)
        dd = [sc(a) - sc(b) for a, b in zip(R[nm], ref)]
        mix1 = sc(0.225 * p + 0.525 * tab + 0.25 * din) - b1
        mix2 = sc(0.40 * p + 0.20 * tab + 0.40 * din) - b2
        print("  %-14s %8.1f   %s %d/3   %+9.1f %+10.1f"
              % (nm, sc(p), " ".join("%+7.1f" % v for v in dd),
                 sum(1 for v in dd if v > 0), mix1, mix2), flush=True)
    print("\n  통과분만 VS=2022 로 확정한다. 하나씩 넣었고 묶지 않았다.")
