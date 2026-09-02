# -*- coding: utf-8 -*-
"""남은 두 피처 후보를 plat 고친 기준선 위에서, 두 폴드로 잰다.

왜 이 둘만 남았나
    bdev_ahead     타자 쪽이 비대칭이다. 투수 관련 열이 16개인데 타자는 3개다
                   (asof_batter_n / success_rate / middle_rate). 제구가 투수의
                   일이라 이해는 되지만 5배 차이다.
                   조건부 스크린에서 퓨처스 +0.0093 으로 통과했는데 관문에
                   넣어본 적이 없다.

    fb_rate_gap    팀원 트랙맨 27피처 중 유일하게 통과한 것.
                   편상관 -0.0090 (전체) / -0.0133 (저표본), 귀무 0.00093 의
                   10~14배. 커버리지 93.9%. 카운트 조건부 구종선택이라
                   44열에 없는 축이다. 스크린만 하고 계속 미뤘다.

경고를 미리 적어둔다
    피처 추가는 관문이 여섯 번 틀린 부류다. 그리고 오늘 전이율 표에서
    개선 후보는 전부 0.08~0.16 이거나 부호가 뒤집혔다.

    다만 오늘 유일하게 통한 plat_dev 도 결국 피처였다. 차이는 그게 '새 정보'
    가 아니라 '틀린 계산의 수정' 이었다는 것이다. 이 둘은 새 정보라서
    성격이 다르고, 기대치를 낮게 잡아야 한다.

    그래서 판정선을 높인다 — **두 폴드에서 +10 이상 & 3/3**.

팔 (기준선은 plat_dev as-of, submit_30 과 같은 구성)
    base        현행
    bdev        + bdev_ahead
    fbgap       + fb_rate_gap
    both        둘 다   (겹치는지 별개인지)

    '통과분은 하나씩 넣어라' 규율대로 단독을 먼저 보고 both 로 합산성을 본다.
    5개 묶으면 3폴드 전부 음수, 최강 1개만 넣으면 전부 양수였던 전례가 있다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, "/workspace/aimers")
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
DL = "/workspace/aimers/colab/_dl"
SEEDS = (42, 1, 777)
LR1 = 3e-3
OLD_F_MAX = 2022
OLD_W = 0.1
EP2 = {"all": 1, "regular": 1, "futures": 4}
FOLDS = (2022, 2024)
ARMS = (("base", []), ("bdev", ["bdev_ahead"]),
        ("fbgap", ["fb_rate_gap"]), ("both", ["bdev_ahead", "fb_rate_gap"]))

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402


def one(seed, tr_idx, t_isf, w, season, VS, gate, isf_g):
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("regular", ~t_isf), ("futures", t_isf)):
        idx, ww = tr_idx[sel], w[sel]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model()
        G.train(m, idx, 2, LR1, w=ww, seed=seed, tag=f"VS{VS} {br} s{seed} S1")
        s2 = idx[season[idx] == VS - 1]
        pr = G.stage2_params(m)
        for e in range(EP2[br]):
            G.train(m, s2, 1, 2e-4, params=pr, seed=seed + e,
                    tag=f"VS{VS} {br} s{seed} S2e{e+1}")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    return np.where(isf_g, 0.6 * P["all"] + 0.4 * P["futures"],
                    0.6 * P["all"] + 0.4 * P["regular"])


if __name__ == "__main__":
    d0 = F.build(DATA, VS=2024)
    season = d0["season"].astype(np.float64)
    isf, y = d0["is_f"], d0["y"].astype(np.float64)
    F44 = list(d0["F44"])
    raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                      usecols=["row_id", "pitcher_id"])

    # ---- 후보 두 열을 train.csv 행 순서로 맞춰 준비
    cond = pd.read_csv(os.path.join(DL, "cond_pass.csv.gz"),
                       usecols=["row_id", "bdev_ahead"])
    assert (cond["row_id"].to_numpy() == raw["row_id"].to_numpy()).all(), \
        "cond_pass 행 순서가 train.csv 와 다르다"
    tm27 = pd.read_csv(os.path.join(DL, "tm27_pass.csv.gz"),
                       usecols=["pitcher_id", "fb_rate_gap"])
    EXTRA = {
        "bdev_ahead": cond["bdev_ahead"].to_numpy(np.float32),
        "fb_rate_gap": raw["pitcher_id"].map(
            dict(zip(tm27["pitcher_id"], tm27["fb_rate_gap"]))
        ).to_numpy(np.float64),
    }
    cov = float(np.isfinite(EXTRA["fb_rate_gap"]).mean())
    EXTRA["fb_rate_gap"] = np.nan_to_num(EXTRA["fb_rate_gap"],
                                         nan=0.0).astype(np.float32)
    print(f"  fb_rate_gap 커버리지 {cov*100:.1f}%  (없는 투수는 0 = 중립)")

    tr_sorted = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                              encoding="utf-8-sig"))
    pos = pd.Series(np.arange(len(tr_sorted)),
                    index=tr_sorted["row_id"].to_numpy())
    pos_map = pos.reindex(raw["row_id"].to_numpy()).to_numpy()

    ALL = {}
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        G.gate, G.yv = gate, yv
        m_tr = season < VS

        hist = PP.fit_history_tables(tr_sorted[tr_sorted.season < VS])
        Xs = PP.transform_features(tr_sorted, hist, train_mode=True)
        cols = list(Xs.columns)
        Xfr = Xs.to_numpy(dtype=np.float32)[pos_map]
        ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
        dv = F.build(DATA, VS=VS)
        Xfr[:, cols.index("plat_dev")] = \
            dv["X44"][:, F44.index("plat_dev")].astype(np.float32)

        old = season <= OLD_F_MAX
        c4 = np.where(old & isf, 0.0,
                      np.where(old & ~isf, 1.0,
                               np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
        tr_idx = np.where(m_tr)[0]
        t_isf = isf[tr_idx]
        w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
        G.log(f"\n  VS={VS}  학습 {len(tr_idx):,}  검증 {len(gate):,}")

        for nm, adds in ARMS:
            cols_extra = [EXTRA[a][:, None] for a in adds]
            Xin = np.concatenate([Xfr, c4] + cols_extra, 1)
            Xn, Xc, cards = G.prep(Xin, m_tr, ci + [Xfr.shape[1]])
            G.Xn, G.cards = Xn, cards
            G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
            t0 = time.time()
            ps = [one(sd, tr_idx, t_isf, w10, season, VS, gate, isf_g)
                  for sd in SEEDS]
            for i, sd in enumerate(SEEDS):
                np.save(os.path.join(OUT, f"lf_{VS}_{nm}_s{sd}.npy"), ps[i])
            ALL[(VS, nm)] = ps
            G.log(f"    {nm:6s} (+{len(adds)}열) 끝 {time.time()-t0:.0f}s")

    def sc(p, yv, m=None):
        m = np.ones(len(yv), bool) if m is None else m
        return F.best_shift(p[m], yv[m])[0]

    print("\n" + "=" * 88)
    print("  남은 피처 후보 — plat 고친 기준선 위, 두 폴드")
    print("=" * 88)
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        base = ALL[(VS, "base")]
        print(f"\n  VS={VS}")
        for nm, adds in ARMS:
            ps = ALL[(VS, nm)]
            p = np.mean(ps, 0)
            line = (f"    {nm:6s} 전체 {sc(p, yv):8.1f}  1군 {sc(p, yv, ~isf_g):8.1f}"
                    f"  퓨처스 {sc(p, yv, isf_g):8.1f}")
            if nm != "base":
                dd = [sc(a, yv) - sc(b, yv) for a, b in zip(ps, base)]
                mu = float(np.mean(dd))
                se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
                line += (f"   차이 {mu:+6.1f}+-{se:4.1f} t={mu/max(se,1e-9):5.2f}"
                         f" {sum(1 for x in dd if x>0)}/{len(dd)}")
            print(line)

    print("\n  판정선 — 두 폴드에서 +10 이상 & 3/3.")
    print("  피처 추가는 관문이 여섯 번 틀린 부류라 기준을 높게 잡는다.")
    print("  both 가 단독들의 합보다 작으면 겹치는 것이다.")
    print("  둘 다 지면 44열로 짜낼 건 다 짰다는 결론이 확정된다.")
