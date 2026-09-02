# -*- coding: utf-8 -*-
"""TabM 에 시즌가중을 건다. 이 축은 절반만 되어 있었다.

현황
    CatBoost   시즌가중 2.0^(season-2019)   배치됨
    HistGB     시즌가중 2.0                 배치됨
    TabM       decay=0.0                    **없음**

    혼합의 70~80% 를 지는 TabM 의 Stage1 이 2019~2024 를 균일하게 본다.
    2019년 투구와 2024년 투구가 같은 무게다. export_final.py 가 모델 메타에
    decay=0.0 을 그대로 적고 있다.

    기록에는 '시즌가중 -> CatBoost 996->1021, TabM 관문 +28' 이 남아 있는데
    CatBoost 에만 실리고 TabM 에는 안 실렸다.

이 축은 관문을 믿어도 된다
    관문이 맞히는 부류    스케줄러 / 시즌가중 / lr      = 최적화·학습분포
    관문이 틀린 부류      브랜치 / 계열 / 에폭 / 피처    = 조합 (여섯 번 틀림)
    시즌가중은 앞쪽이다. 기전도 실재한다 — 2023 ABS 로 퓨처스 성공률이
    70.9% -> 47.3% 로 꺾였다. 분포가 실제로 이동한다.

미리 짚을 두 가지
    ① Stage2 와 중복일 수 있다. 마지막 시즌으로 미세조정하는 것과 마지막
       시즌에 가중을 주는 게 같은 일을 할 수 있다. 그래서 Stage2 를 **켠 채로**
       얹었을 때의 증분으로 잰다. decay 단독 성능이 아니다.
    ② CatBoost 의 2.0 은 공격적이다 (2024 가 2019 의 32배). TabM 은 2에폭이라
       더 완만한 값이 맞을 수 있어 훑는다.

팔
    d0.0        현행 (OLD_W 0.1 만)
    d1.2 d1.5 d2.0   시즌가중 x OLD_W 0.1
    d1.5_noold  시즌가중만, OLD_W 없음
                -> 옛퓨처스 0.1 은 2단계 계단인데, 매끄러운 감쇠가 그걸
                   흡수하는지 본다. 흡수하면 장치 하나를 줄일 수 있다.
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
#      (이름, decay, 옛퓨처스 가중)
ARMS = (("d0.0", 0.0, OLD_W), ("d1.2", 1.2, OLD_W), ("d1.5", 1.5, OLD_W),
        ("d2.0", 2.0, OLD_W), ("d1.5_noold", 1.5, 1.0))

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
R = ~is_f
FULL = np.ones(len(gate), bool)


def sc(p, m=None):
    m = FULL if m is None else m
    return F.best_shift(p[m], yv[m])[0]


def one(seed, tr_idx, t_isf, w, season):
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("regular", ~t_isf), ("futures", t_isf)):
        idx, ww = tr_idx[sel], w[sel]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model()
        G.train(m, idx, 2, LR1, w=ww, seed=seed, tag=f"{br} s{seed} S1")
        s2 = idx[season[idx] == VS - 1]
        ps = G.stage2_params(m)
        for e in range(EP2[br]):
            G.train(m, s2, 1, 2e-4, params=ps, seed=seed + e,
                    tag=f"{br} s{seed} S2e{e+1}")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    return (0.6 * P["all"] + 0.4 * P["regular"],
            0.6 * P["all"] + 0.4 * P["futures"])


if __name__ == "__main__":
    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr[tr.season < VS])
    Xs = PP.transform_features(tr, hist, train_mode=True)
    cols = list(Xs.columns)
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                         usecols=["row_id"])
    pos = pd.Series(np.arange(len(tr)), index=tr["row_id"].to_numpy())
    Xf = Xs.to_numpy(dtype=np.float32)[
        pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()]
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]

    season, isf = G.season.astype(np.float64), G.is_f
    old = season <= OLD_F_MAX
    c4 = np.where(old & isf, 0.0,
                  np.where(old & ~isf, 1.0,
                           np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
    Xn, Xc, cards = G.prep(np.concatenate([Xf, c4], 1), G.m_tr,
                           ci + [Xf.shape[1]])
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)

    tr_idx = np.where(G.m_tr)[0]
    t_isf = isf[tr_idx]
    s_tr = season[tr_idx]
    G.log(f"  학습 {len(tr_idx):,}   Stage2 는 그대로 켠 채로 잰다 "
          f"(all/regular 1에폭, futures 4에폭)")
    G.log(f"  {'팔':12s} {'2019':>7s} {'2024':>7s}  상대무게")
    for nm, dk, ow in ARMS:
        base = np.ones(len(tr_idx)) if dk == 0.0 else dk ** (s_tr - 2019)
        G.log(f"  {nm:12s} {base[s_tr==2019][0]:7.2f} "
              f"{base[s_tr==2024][0] if (s_tr==2024).any() else float('nan'):7.2f}"
              f"   옛퓨처스 x{ow}")

    RES = {}
    for nm, dk, ow in ARMS:
        w = np.where(t_isf & old[tr_idx], ow, 1.0).astype(np.float64)
        if dk > 0.0:
            w = w * (dk ** (s_tr - 2019))
        t0 = time.time()
        RES[nm] = [one(sd, tr_idx, t_isf, w, season) for sd in SEEDS]
        for i, sd in enumerate(SEEDS):
            np.save(os.path.join(OUT, f"sd_{nm}_r_s{sd}.npy"), RES[nm][i][0])
            np.save(os.path.join(OUT, f"sd_{nm}_f_s{sd}.npy"), RES[nm][i][1])
        G.log(f"  {nm:12s} 끝 {time.time()-t0:.0f}s")

    def full(x):
        return np.where(is_f, x[1], x[0])

    G.log("\n" + "=" * 90)
    G.log(f"  {'팔':12s} {'전체':>8s} {'1군행':>8s} {'퓨처스행':>9s}"
          f"   현행 대비 (짝차이, 부호일치)")
    G.log("=" * 90)
    base = RES["d0.0"]
    for nm, *_ in ARMS:
        r = RES[nm]
        v = [float(np.mean([sc(full(x)) for x in r])),
             float(np.mean([sc(x[0], R) for x in r])),
             float(np.mean([sc(x[1], is_f) for x in r]))]
        if nm == "d0.0":
            G.log(f"  {nm:12s} {v[0]:8.1f} {v[1]:8.1f} {v[2]:9.1f}   <- 현행")
            continue
        tail = []
        for lab, fn, msk in (("전체", full, FULL),
                             ("1군", lambda x: x[0], R),
                             ("퓨처스", lambda x: x[1], is_f)):
            d = [sc(fn(a), msk) - sc(fn(b), msk) for a, b in zip(r, base)]
            mu = float(np.mean(d))
            se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
            tail.append(f"{lab} {mu:+6.1f}+-{se:4.1f} "
                        f"{sum(1 for x in d if x > 0)}/{len(d)}")
        G.log(f"  {nm:12s} {v[0]:8.1f} {v[1]:8.1f} {v[2]:9.1f}   "
              + "  ".join(tail))

    G.log("\n  읽는 법")
    G.log("    이 축은 관문을 믿는다 — 최적화·학습분포 부류이고 기전이 실재한다.")
    G.log("    고원이 보이면 덜 극단적인 값을 고른다 (Stage2 에폭 때와 같은 규율).")
    G.log("    d1.5_noold 가 d1.5 와 비슷하면 OLD_W 계단을 없앨 수 있다.")
    G.log("    전부 음수면 Stage2 가 이미 같은 일을 하고 있다는 뜻이다.")
