# -*- coding: utf-8 -*-
"""다양성원을 **추가 멤버**로 재본다. 지금까지는 전부 대체재로만 쟀다.

왜 이 각도인가
    시드 평균이 리더보드에서 실패했다 (1058 -> 1056). 원인을 보니
        두 예측의 상관 0.997,  행별 차이 평균 0.00194 (예측 SD 의 6%)
    시드는 다양성원으로 너무 약하다. 더 큰 다양성이 필요하다.

    그리고 MNCA 가 증명한다 — 단독 901 로 TabM(906)보다 낮은데 얹으면 +6.6 이다.
    **개별로 지는 모델이 앙상블에는 기여할 수 있다.** 우리는 지금까지 후보를
    전부 '현행을 대체할 수 있나' 로만 쟀다.

한 번도 안 재본 것 넷
    ourfeat   우리 features44 로 학습한 TabM
              지인 전처리와 파생 열이 다르다. 기억에 '전처리 통일하니 상관
              0.894 -> 0.924 로 올라 손해봤다' 가 남아 있는데, 그때는 대체재로
              재서 졌다. 낮은 비중의 추가 멤버로는 안 재봤다.
    k64       어댑터 32 -> 64
    d512      블록 폭 256 -> 512
    b4        블록 3 -> 4
              셋 다 용량 스윕에서 '대체하면 진다' 로 닫았다. 추가는 안 봤다.

판정
    기준 = 0.30 CatBoost + 0.70 TabM8(지인피처).  거기에 비중 a 로 얹는다.
    시드별 짝차이로 오차막대를 낸다 — MNCA 에서 3시드평균 예측 하나로 재서
    +9.0 이 나왔다가 제대로 재니 +6.6 이었던 전례가 있다.

    그리고 관문이 조합 축에서 12배 과대평가한다는 걸 실측했다
    (w_CB 0.10->0.30 이 관문 +12.1 / 리더보드 +1.0). 그러니 관문 +5 는
    리더보드 +0.4 쯤이다. **관문 +15 이상 & 3/3 이 아니면 안 옮긴다.**
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
SEEDS8 = (42, 1, 777, 2, 7, 13, 99, 2024)
#      (이름, 피처세트, make_model 인자)
ARMS = (("ourfeat", "ours", {}),
        ("k64", "friend", dict(k=64)),
        ("d512", "friend", dict(d_block=512)),
        ("b4", "friend", dict(n_blocks=4)))

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


def one(seed, tr_idx, t_isf, w, season, mk):
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("regular", ~t_isf), ("futures", t_isf)):
        idx, ww = tr_idx[sel], w[sel]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model(**mk)
        G.train(m, idx, 2, LR1, w=ww, seed=seed, tag=f"{br} s{seed} S1")
        s2 = idx[season[idx] == VS - 1]
        ps = G.stage2_params(m)
        for e in range(EP2[br]):
            G.train(m, s2, 1, 2e-4, params=ps, seed=seed + e,
                    tag=f"{br} s{seed} S2e{e+1}")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    return np.where(is_f, 0.6 * P["all"] + 0.4 * P["futures"],
                    0.6 * P["all"] + 0.4 * P["regular"])


if __name__ == "__main__":
    # ---- 두 벌의 피처를 준비한다
    d44 = F.build(DATA, VS=VS)
    season, isf = G.season.astype(np.float64), G.is_f
    old = season <= OLD_F_MAX
    c4 = np.where(old & isf, 0.0,
                  np.where(old & ~isf, 1.0,
                           np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]

    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr[tr.season < VS])
    Xs = PP.transform_features(tr, hist, train_mode=True)
    cols = list(Xs.columns)
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                         usecols=["row_id"])
    pos = pd.Series(np.arange(len(tr)), index=tr["row_id"].to_numpy())
    Xfr = Xs.to_numpy(dtype=np.float32)[
        pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()]
    ci_fr = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]

    PACK = {
        "friend": G.prep(np.concatenate([Xfr, c4], 1), G.m_tr,
                         ci_fr + [Xfr.shape[1]]),
        "ours": G.prep(np.concatenate([d44["X44"].astype(np.float32), c4], 1),
                       G.m_tr, list(d44["cat_idx"]) + [d44["X44"].shape[1]]),
    }
    for k, v in PACK.items():
        G.log(f"  {k:7s} 수치 {v[0].shape[1]}열  범주 {len(v[2])}열")

    tr_idx = np.where(G.m_tr)[0]
    t_isf = isf[tr_idx]
    w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)

    TM8 = np.mean([np.load(f"{OUT}/sc_s{s}.npy") for s in SEEDS8], 0)
    REF = 0.30 * G.CB + 0.70 * TM8
    r0 = sc(REF)
    G.log(f"\n  기준 (CB 0.30 / TabM8 0.70)  {r0:.1f}   "
          f"TabM8 단독 {sc(TM8):.1f}\n")

    AL = (0.05, 0.10, 0.15, 0.20, 0.30)
    for nm, feat, mk in ARMS:
        Xn, Xc, cards = PACK[feat]
        G.Xn, G.cards = Xn, cards
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
        t0 = time.time()
        try:
            ps = [one(sd, tr_idx, t_isf, w10, season, mk) for sd in SEEDS]
        except Exception as e:
            G.log(f"  {nm} 실패: {type(e).__name__} {str(e)[:140]}")
            continue
        for i, sd in enumerate(SEEDS):
            np.save(os.path.join(OUT, f"div_{nm}_s{sd}.npy"), ps[i])
        p = np.mean(ps, 0)
        cells = []
        for a in AL:
            dd = [sc((1 - a) * REF + a * q) - r0 for q in ps]
            mu = float(np.mean(dd))
            se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
            cells.append((a, mu, se, sum(1 for x in dd if x > 0)))
        best = max(cells, key=lambda c: c[1])
        G.log(f"  {nm:8s} 단독 {sc(p):7.1f}  1군 {sc(p, R):7.1f}  "
              f"퓨처스 {sc(p, is_f):7.1f}  기준상관 {np.corrcoef(p, REF)[0,1]:.4f}"
              f"  {time.time()-t0:5.0f}s")
        G.log(f"  {'':8s} " + "  ".join(f"a{a:.2f} {mu:+5.1f}+-{se:4.1f}"
                                        for a, mu, se, _ in cells)
              + f"   최적 a={best[0]:.2f} {best[1]:+.1f} "
                f"t={best[1]/max(best[2],1e-9):.2f} {best[3]}/{len(ps)}")

    G.log("\n  판정선 — 관문이 조합 축에서 12배 과대평가한다 (실측).")
    G.log("  관문 +15 이상 & 3/3 이 아니면 안 옮긴다. MNCA(+6.6)도 그 기준엔 못 미친다.")
    G.log("  '기준상관' 이 낮을수록 다양성원으로서 값어치가 있다 —")
    G.log("  시드 평균은 상관 0.997 이라 실패했다.")
