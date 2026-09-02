# -*- coding: utf-8 -*-
"""ball 의 **플래툰/주자상황 편차**를 관문에 올린다. reverse 와는 다른 축이다.

왜 이걸 재나
    라벨 5종 x 상황축 5종 = 25칸을 44열 통제 편상관으로 훑었다. 통과는 3칸뿐.
        reverse_bh  -0.00771     ball_bh  +0.00606     ball_base  +0.00539
    카운트/이닝/아웃 축 15칸은 전부 임계(0.005) 미달이다. 축은 타자손 하나뿐이었다.

    그리고 reverse_bh 는 **관문에서 졌다** (VS=2022 +2.7 t=1.49 2/3).
    이유가 상관행렬에 있었다 — plat_dev 와 -0.814 다. 이미 실린 축의 거울상이었다.

        reverse_bh   ball_bh   ball_base   plat_dev
        +1.000        -0.084     -0.007      -0.814
        -0.084        +1.000     +0.120      +0.200
        -0.007        +0.120     +1.000      -0.010

    ball_bh 는 plat_dev 와 0.200, reverse 와 -0.084 다. **새 축이다.**
    ball_base 는 거의 직교다 (-0.010). 그래서 reverse 의 사망이 이 둘을 말해주지 않는다.

라벨의 뜻
    success/reverse/middle 은 제구 판정 분할이고 ball/strike 는 **투구 결과 분할**이다.
    (success .5204 + reverse .2298 + middle .1465, ball .3931 + strike .4569)
    즉 ball 편차는 "이 투수가 이 타자손 상대로 볼을 더 던지나" 로, 제구 성공률
    편차(plat_dev)와 다른 것을 잰다.

팔 — 묶지 않는다
    base       현행 (44열 + c4)
    ball_bh    + ball_bh      가장 센 새 축
    ball_base  + ball_base    가장 독립적인 축
    "5개 묶으면 3폴드 전부 음수, 최강 1개만 넣으면 전부 양수" 를 따른다.

판정선 — 두 폴드에서 +10 이상 & 3/3
    피처 추가는 관문이 여섯 번 틀린 부류다. plat_dev 는 +19.5/+24.2 였다.

열은 colab/_dl/label_family.csv.gz (colab/label_family2.py 가 만든다).
season 단위 shift(1) 누적이라 폴드별로 as-of 다. 리그 사전값만 전 구간인데
그건 이미 "리그 사전값은 누출이 아니다" 로 확인됐다 (투수당 열만 진짜였다).
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
OLD_F_MAX, OLD_W = 2022, 0.1
EP2 = {"all": 1, "regular": 1, "futures": 4}
FOLDS = (2022, 2024)
ARMS = (("base", []), ("ball_bh", ["ball_bh"]),
        ("ball_base", ["ball_base"]))

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
                      usecols=["row_id"])
    rid = raw["row_id"].to_numpy()

    tr_sorted = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                              encoding="utf-8-sig"))
    pos = pd.Series(np.arange(len(tr_sorted)),
                    index=tr_sorted["row_id"].to_numpy())
    pos_map = pos.reindex(rid).to_numpy()

    ALL = {}
    for VS in FOLDS:
        # ---- 그 폴드의 as-of 편차표. row_id 로 명시 정렬한다
        pcd = pd.read_csv(os.path.join(DL, "label_family.csv.gz")
                          ).set_index("row_id").reindex(rid)
        if pcd.isna().any().any():
            raise ValueError("label_family row_id 정렬 실패")
        EX = {c: pcd[c].to_numpy(np.float32) for c in pcd.columns}
        G.log(f"\n  VS={VS}  편차열 {list(EX)}  "
              f"비영 {float((EX['ball_bh']!=0).mean())*100:.1f}%")

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
        G.log(f"  학습 {len(tr_idx):,}  검증 {len(gate):,}")

        for nm, adds in ARMS:
            Xin = np.concatenate([Xfr, c4] + [EX[a][:, None] for a in adds], 1)
            Xn, Xc, cards = G.prep(Xin, m_tr, ci + [Xfr.shape[1]])
            G.Xn, G.cards = Xn, cards
            G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
            t0 = time.time()
            ps = [one(sd, tr_idx, t_isf, w10, season, VS, gate, isf_g)
                  for sd in SEEDS]
            for i, sd in enumerate(SEEDS):
                np.save(os.path.join(OUT, f"bg_{VS}_{nm}_s{sd}.npy"), ps[i])
            ALL[(VS, nm)] = ps
            G.log(f"    {nm:7s} (+{len(adds)}열) 끝 {time.time()-t0:.0f}s")

    def sc(p, yv, m=None):
        m = np.ones(len(yv), bool) if m is None else m
        return F.best_shift(p[m], yv[m])[0]

    print("\n" + "=" * 92)
    print("  ball 편차 (라벨 복원) — 하나씩, 두 폴드")
    print("=" * 92)
    for VS in FOLDS:
        gate = np.where(season == VS)[0]
        isf_g, yv = isf[gate], y[gate]
        base = ALL[(VS, "base")]
        print(f"\n  VS={VS}")
        for nm, adds in ARMS:
            ps = ALL[(VS, nm)]
            p = np.mean(ps, 0)
            line = (f"    {nm:7s} 전체 {sc(p, yv):8.1f}  1군 {sc(p, yv, ~isf_g):8.1f}"
                    f"  퓨처스 {sc(p, yv, isf_g):8.1f}")
            if nm != "base":
                dd = [sc(a, yv) - sc(b, yv) for a, b in zip(ps, base)]
                mu = float(np.mean(dd))
                se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
                line += (f"   차이 {mu:+6.1f}+-{se:4.1f} t={mu/max(se,1e-9):5.2f}"
                         f" {sum(1 for x in dd if x>0)}/{len(dd)}")
            print(line)
    print("\n  판정선 두 폴드 +10 & 3/3. plat_dev 는 +19.5/+24.2 였다.")
    print("  bundle 이 pcbr 보다 나쁘면 '묶지 말라' 는 규율이 또 확인된 것이다.")
