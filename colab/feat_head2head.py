# -*- coding: utf-8 -*-
"""피처셋 정면 비교. div_add 에 기준 팔이 없어서 다시 잰다.

div_add 가 왜 부족했나
    ourfeat 930.0   우리 피처 + 기본 구조 (k32/d256/b3)
    k64     893.5   지인 피처 + k64      <- 구조가 다르다
    d512    900.2   지인 피처 + d512     <- 구조가 다르다
    b4      906.5   지인 피처 + blocks4  <- 구조가 다르다

    **같은 구조에 지인 피처인 팔이 없었다.** 906 은 b4 에서 유추한 값이다.
    그리고 비교 대상으로 쓴 TM8 은 8시드인데 ourfeat 은 3시드였다.
    폴드도 VS=2024 하나뿐이다.

이번 설계 — 바뀌는 것은 피처셋 하나
    두 팔     friend(지인 preprocess)  vs  ours(features44)
    고정      구조 k32/d256/b3, 시드 3개, 라우팅 0.6/0.4,
              Stage1 2에폭 lr 3e-3, Stage2, 옛퓨처스 가중 0.1, 같은 학습 코드
    두 폴드   VS=2023, VS=2024
              한 폴드에서만 이기면 그 해 과적합을 의심한다.

누출은 확인했다
    features44 의 룩업 4열(lg_cm_eff, cm_rel, p_adj_cm, plat_dev)이 VS 를 바꾸면
    값이 달라진다 -> 학습 창이 제대로 걸려 있다. 나머지 열은 주어진 asof_* 이거나
    그로부터 as-of 로 파생된 것이다.

왜 중요한가
    사실이면 지금 배치가 **더 나쁜 피처셋을 쓰고 있다**는 뜻이다. 다만 우리 피처로
    학습하려면 우리 학습 코드를 써야 하는데 그건 리더보드에서 지인 코드보다
    4~5점 낮았다 (1052 vs 1058). 피처 이득이 그 손해를 넘어야 옮길 값어치가 있다.
    그래서 크기를 정확히 알아야 한다.
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
SEEDS = (42, 1, 777)
LR1 = 3e-3
OLD_F_MAX = 2022
OLD_W = 0.1
EP2 = {"all": 1, "regular": 1, "futures": 4}
FOLDS = (2022, 2024)   # 2023 은 퓨처스 체제전환년이라 폴드가 깨진다

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402


def make_packs(VS, tr_sorted, pos_map, season, isf):
    """두 피처셋을 같은 창(season < VS)으로 만든다."""
    old = season <= OLD_F_MAX
    c4 = np.where(old & isf, 0.0,
                  np.where(old & ~isf, 1.0,
                           np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
    m_tr = season < VS

    hist = PP.fit_history_tables(tr_sorted[tr_sorted.season < VS])
    Xs = PP.transform_features(tr_sorted, hist, train_mode=True)
    cols = list(Xs.columns)
    Xfr = Xs.to_numpy(dtype=np.float32)[pos_map]
    ci_fr = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]

    d = F.build(DATA, VS=VS)             # 룩업이 season < VS 로 걸린다
    Xou = d["X44"].astype(np.float32)
    ci_ou = list(d["cat_idx"])

    return {
        "friend": G.prep(np.concatenate([Xfr, c4], 1), m_tr,
                         ci_fr + [Xfr.shape[1]]),
        "ours": G.prep(np.concatenate([Xou, c4], 1), m_tr,
                       ci_ou + [Xou.shape[1]]),
    }


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
    isf = d0["is_f"]
    y = d0["y"].astype(np.float64)
    tr_sorted = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                              encoding="utf-8-sig"))
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                         usecols=["row_id"])
    pos = pd.Series(np.arange(len(tr_sorted)),
                    index=tr_sorted["row_id"].to_numpy())
    pos_map = pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()

    ALL = {}
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g = isf[gate]
        yv = y[gate]
        G.gate, G.yv = gate, yv
        PACK = make_packs(VS, tr_sorted, pos_map, season, isf)
        tr_idx = np.where(season < VS)[0]
        t_isf = isf[tr_idx]
        old = season <= OLD_F_MAX
        w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
        G.log(f"\n  VS={VS}  학습 {len(tr_idx):,}  검증 {len(gate):,} "
              f"(1군 {int((~isf_g).sum()):,})")

        for feat in ("friend", "ours"):
            Xn, Xc, cards = PACK[feat]
            G.Xn, G.cards = Xn, cards
            G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
            t0 = time.time()
            ps = [one(sd, tr_idx, t_isf, w10, season, VS, gate, isf_g)
                  for sd in SEEDS]
            for i, sd in enumerate(SEEDS):
                np.save(os.path.join(OUT, f"h2h_{VS}_{feat}_s{sd}.npy"), ps[i])
            ALL[(VS, feat)] = ps
            G.log(f"    {feat:7s} 끝 {time.time()-t0:.0f}s")

    def sc(p, yv, m=None):
        m = np.ones(len(yv), bool) if m is None else m
        return F.best_shift(p[m], yv[m])[0]

    print("\n" + "=" * 86)
    print("  피처셋 정면 비교 — 구조·시드·학습코드 전부 고정, 피처만 다름")
    print("=" * 86)
    print(f"  {'폴드':>6s} {'피처':>8s} {'전체':>9s} {'1군':>9s} {'퓨처스':>9s}"
          f"   시드별(전체)")
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        for feat in ("friend", "ours"):
            ps = ALL[(VS, feat)]
            p = np.mean(ps, 0)
            solos = [sc(q, yv) for q in ps]
            print(f"  {VS:6d} {feat:>8s} {sc(p, yv):9.1f} "
                  f"{sc(p, yv, ~isf_g):9.1f} {sc(p, yv, isf_g):9.1f}   "
                  + "[" + ", ".join(f"{v:.0f}" for v in solos) + "]")
        # 시드 짝차이
        dd = [sc(a, yv) - sc(b, yv)
              for a, b in zip(ALL[(VS, "ours")], ALL[(VS, "friend")])]
        mu = float(np.mean(dd))
        se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
        print(f"  {'':6s} {'차이':>8s} {mu:+9.1f} +- {se:.1f}  "
              f"t={mu/max(se,1e-9):.2f}  {sum(1 for x in dd if x>0)}/{len(dd)}"
              f"   개별 [{', '.join(f'{v:+.0f}' for v in dd)}]")
        print(f"  {'':6s} {'상관':>8s} "
              f"{np.corrcoef(np.mean(ALL[(VS,'ours')],0), np.mean(ALL[(VS,'friend')],0))[0,1]:9.4f}")

    print("\n  두 폴드에서 같은 부호로 크게 이겨야 믿는다.")
    print("  한 폴드에서만 이기면 그 해 과적합이다 — 다년 확인이 이 실험의 핵심이다.")
    print("  이득이 5점 미만이면 옮길 값어치가 없다 — 우리 학습 코드가 지인보다")
    print("  리더보드에서 4~5점 낮기 때문이다 (1052 vs 1058).")
