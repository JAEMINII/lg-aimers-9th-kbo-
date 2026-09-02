# -*- coding: utf-8 -*-
"""plat_dev 한 열만 as-of 로 바꿔서 +23.8 이 재현되는지 본다.

발견
    두 전처리가 만드는 44열을 대조했더니 **plat_dev 하나만** 다르다.
        두 판본 상관 0.774,  최대차 0.089
        나머지 43열은 상관 1.00000, 최대차 0.0000

    그리고 계산 방식이 이렇게 갈린다.
        지인  pid.map(history["plat_l"])   투수당 값 하나. 학습창 전체로 계산
              -> 2019년 행이 2020~2023 결과를 본다. **학습 중 미래 누출**
        우리  PC.cumsum().shift(1)         (투수, 시즌)별 누적을 한 시즌 밀어냄
              -> 2021년 행은 2019~2020 만 본다. as-of

    증상도 누출과 일치한다.
        지인 plat_dev  표적상관 0.021 (높음)   <- 누출로 부풀려짐
        우리 plat_dev  표적상관 0.013 (낮음)
        검증 점수      우리가 +23.8 (3/3, t=6.73)
    학습 상관이 높은데 검증에서 지는 건 누출의 교과서적 증상이다.

왜 이 실험이 중요한가
    사실이면 파이프라인을 통째로 갈 필요가 없다. 원래는 우리 전처리로 바꾸려면
    우리 학습 코드를 써야 해서 -4~5점을 물어야 했다 (리더보드 1052 vs 1058).
    한 열만 고치면 그 비용이 0 이다.

    그리고 추론 경로는 안 건드려도 된다 — 2025 시험 행에게는 '2019~2024 전부'가
    정당한 과거라 지금 고정 테이블이 이미 맞다. 바뀌는 건 학습 행의 피처뿐이다.

세 팔 (지인 피처 고정, plat_dev 만 교체)
    friend    지인 plat_dev  (현행)
    asof      우리 as-of plat_dev
    drop      plat_dev 를 상수 0 으로  (열을 아예 죽였을 때의 바닥)

    drop 을 넣는 이유 — asof 가 이기는 게 '더 좋은 피처' 때문인지 '나쁜 피처를
    없앤 것' 때문인지 가른다. drop 이 friend 보다 이미 이기면 지인 판본은
    있는 것보다 없는 게 나은 열이라는 뜻이다.

두 폴드 (VS=2022, 2024)
    한 폴드에서만 이기면 그 해 과적합이다. VS=2023 은 퓨처스 체제전환년이라 쓰지 않는다.
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
FOLDS = (2022, 2024)

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
        isf_g, yv = isf[gate], y[gate]
        G.gate, G.yv = gate, yv
        m_tr = season < VS

        hist = PP.fit_history_tables(tr_sorted[tr_sorted.season < VS])
        Xs = PP.transform_features(tr_sorted, hist, train_mode=True)
        cols = list(Xs.columns)
        Xfr = Xs.to_numpy(dtype=np.float32)[pos_map]
        ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
        j_plat = cols.index("plat_dev")

        # 우리 as-of 판본을 같은 창에서 뽑는다
        dv = F.build(DATA, VS=VS)
        asof_plat = dv["X44"][:, F44.index("plat_dev")].astype(np.float32)

        old = season <= OLD_F_MAX
        c4 = np.where(old & isf, 0.0,
                      np.where(old & ~isf, 1.0,
                               np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]

        VAR = {}
        for nm in ("friend", "asof", "drop"):
            Xv = Xfr.copy()
            if nm == "asof":
                Xv[:, j_plat] = asof_plat
            elif nm == "drop":
                Xv[:, j_plat] = 0.0
            VAR[nm] = G.prep(np.concatenate([Xv, c4], 1), m_tr,
                             ci + [Xfr.shape[1]])

        tr_idx = np.where(m_tr)[0]
        t_isf = isf[tr_idx]
        w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
        G.log(f"\n  VS={VS}  학습 {len(tr_idx):,}  검증 {len(gate):,}  "
              f"plat_dev 열 위치 {j_plat}")
        G.log(f"    두 판본 상관 "
              f"{np.corrcoef(Xfr[:, j_plat], asof_plat)[0,1]:.4f}")

        for nm in ("friend", "asof", "drop"):
            Xn, Xc, cards = VAR[nm]
            G.Xn, G.cards = Xn, cards
            G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
            t0 = time.time()
            ps = [one(sd, tr_idx, t_isf, w10, season, VS, gate, isf_g)
                  for sd in SEEDS]
            for i, sd in enumerate(SEEDS):
                np.save(os.path.join(OUT, f"pf_{VS}_{nm}_s{sd}.npy"), ps[i])
            ALL[(VS, nm)] = ps
            G.log(f"    {nm:7s} 끝 {time.time()-t0:.0f}s")

    def sc(p, yv, m=None):
        m = np.ones(len(yv), bool) if m is None else m
        return F.best_shift(p[m], yv[m])[0]

    print("\n" + "=" * 88)
    print("  plat_dev 한 열만 교체 — 나머지 43열과 학습코드는 전부 동일")
    print("=" * 88)
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        print(f"\n  VS={VS}")
        base = ALL[(VS, "friend")]
        for nm in ("friend", "asof", "drop"):
            ps = ALL[(VS, nm)]
            p = np.mean(ps, 0)
            line = (f"    {nm:7s} 전체 {sc(p, yv):8.1f}  1군 {sc(p, yv, ~isf_g):8.1f}"
                    f"  퓨처스 {sc(p, yv, isf_g):8.1f}")
            if nm != "friend":
                dd = [sc(a, yv) - sc(b, yv) for a, b in zip(ps, base)]
                mu = float(np.mean(dd))
                se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
                line += (f"   차이 {mu:+6.1f}+-{se:4.1f} t={mu/max(se,1e-9):5.2f}"
                         f" {sum(1 for x in dd if x>0)}/{len(dd)}")
            print(line)

    print("\n  읽는 법")
    print("    asof 가 두 폴드에서 +20 근처면 누출이 맞고, 한 열만 고치면 된다.")
    print("    drop 이 friend 보다 이미 이기면 지인 판본은 없느니만 못한 열이다.")
    print("    asof > drop 이면 as-of 판본은 그 자체로 쓸모가 있다.")
