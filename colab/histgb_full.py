# -*- coding: utf-8 -*-
"""HistGB 를 '989 를 냈던 설정 그대로' 재서 지금 앙상블에 자리가 있는지 본다.

앞서 잰 판본이 약했다
    나는 1시드, 라우팅 없음, min_samples_leaf=200 으로 재고 831.3 을 얻었다.
    그런데 리더보드 989 를 낸 판본은 10시드 + 60:40 라우팅 + leaf 500 이다.
    같은 모델이 아니다. 약한 판본으로 재고 기각하면 그 기각은 근거가 없다.

    SOLUTION_958.md 의 설정
        max_iter 400  lr 0.05  max_leaf_nodes 15  min_samples_leaf 500
        l2 1.0  early_stopping False   x 시드 10개 -> 확률 평균

    거기에 나중에 CatBoost 를 996 -> 1021 로 올린 시즌가중(2.0)도 얹어 본다.
    HistGB 에는 시즌가중을 걸어 본 적이 없다 — 그 사이에 CatBoost 로 갈아탔다.

리더보드가 이미 답한 것
    HistGB 60+40 = 989,  CatBoost 60+40 = 987.  둘은 동점이었다.
    그러니 '둘 다 넣기' 는 같은 계열을 두 번 넣는 것에 가깝다. 이 실험은
    그 예상이 지금 구성(0.14 CatBoost + 0.86 TabM)에서도 맞는지 확인한다.

판정
    비교는 반드시 **같은 GBDT 총량**에서 한다. 총량을 올리면 관문 점수가
    오르는데, 그건 HistGB 덕이 아니라 관문이 CatBoost 계열을 과대평가하기
    때문이다 (관문 903.2 -> LB 1021 vs 관문 858.3 -> LB 1041).
"""
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DATA = "open (1)/data"
DL = os.path.join(SC, "_dl")
VS = 2024
SEEDS = (42, 1, 777)
FIXED = -0.005

import features44 as F                                          # noqa: E402

HP = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
          min_samples_leaf=500, l2_regularization=1.0, early_stopping=False)


def fit_predict(Xtr, ytr, w, Xva, seed):
    m = HistGradientBoostingClassifier(random_state=seed, **HP)
    m.fit(Xtr, ytr, sample_weight=w)
    return m.predict_proba(Xva)[:, 1].astype(np.float64)


if __name__ == "__main__":
    d = F.build(DATA, VS=VS)
    X, y, m_tr, m_va = d["X44"], d["y"], d["m_tr"], d["m_va"]
    gate = np.where(m_va)[0]
    yv = y[gate].astype(np.float64)
    isf_all = d["is_f"]
    isf = isf_all[gate]
    Xva = X[gate].astype(np.float64)
    tr_idx = np.where(m_tr)[0]
    season = d["season"].astype(np.float64)

    sc = lambda p, m=slice(None): F.best_shift(p[m], yv[m])[0]      # noqa: E731
    scf = lambda p: F.bss(F.shift(p, FIXED), yv)                    # noqa: E731

    print(f"  학습 {len(tr_idx):,}  관문 {len(gate):,}  시드 {SEEDS}")
    print(f"  설정 {HP}\n")

    out = {}
    for decay in (1.0, 2.0):
        t0 = time.time()
        P = {}
        for br, sel in (("all", np.ones(len(tr_idx), bool)),
                        ("futures", isf_all[tr_idx]),
                        ("regular", ~isf_all[tr_idx])):
            idx = tr_idx[sel]
            w = decay ** (season[idx] - 2019)
            ps = [fit_predict(X[idx].astype(np.float64), y[idx].astype(int),
                              w, Xva, sd) for sd in SEEDS]
            P[br] = np.mean(ps, 0)
        # 989 를 만든 라우팅: 0.6 x 전체 + 0.4 x 경기유형별 단독
        rt = np.where(isf, 0.4 * P["futures"] + 0.6 * P["all"],
                      0.4 * P["regular"] + 0.6 * P["all"])
        out[decay] = (P["all"], rt)
        np.save(os.path.join(DL, f"hg_all_d{decay}.npy"), P["all"])
        np.save(os.path.join(DL, f"hg_route_d{decay}.npy"), rt)
        print(f"  시즌가중 {decay}   전체단독 {sc(P['all']):8.1f}   "
              f"라우팅 {sc(rt):8.1f}   (1군 {sc(rt, ~isf):.1f} / "
              f"퓨처스 {sc(rt, isf):.1f})   {time.time()-t0:.0f}s")

    CB = np.load(os.path.join(SC, "cb_gate.npy")).astype(np.float64)
    seeds_t = ["s42", "s1", "s777", "s2"]
    TM = {s: np.load(os.path.join(DL, f"fb_base_{s}.npy")) for s in seeds_t}
    best = out[2.0][1]
    print(f"\n  CatBoost(배치) {sc(CB):8.1f}    HistGB 최선 {sc(best):8.1f}")
    print(f"  상관  HistGB~CatBoost {np.corrcoef(best, CB)[0,1]:.4f}   "
          f"HistGB~TabM(s42) {np.corrcoef(best, TM['s42'])[0,1]:.4f}   "
          f"CatBoost~TabM(s42) {np.corrcoef(CB, TM['s42'])[0,1]:.4f}")

    print("\n  같은 GBDT 총량에서 대조 (TabM 1시드마다 계산해 평균)")
    print(f"  {'총량':>6s}  {'구성':26s} {'고정':>8s} {'최적':>8s} {'차이':>8s}  시드별")
    for tot in (0.14, 0.20, 0.29, 0.40):
        ref = [scf((1 - tot) * TM[s] + tot * CB) for s in seeds_t]
        refo = [sc((1 - tot) * TM[s] + tot * CB) for s in seeds_t]
        print(f"  {tot:6.2f}  {'CatBoost 단독':26s} {np.mean(ref):8.1f} "
              f"{np.mean(refo):8.1f} {0.0:+8.1f}")
        for nm, frac in (("+HistGB 반반", 0.5), ("HistGB 로 전부 교체", 1.0)):
            v, o, dd = [], [], []
            for i, s in enumerate(seeds_t):
                q = ((1 - tot) * TM[s] + tot * (1 - frac) * CB
                     + tot * frac * best)
                v.append(scf(q)); o.append(sc(q)); dd.append(v[-1] - ref[i])
            print(f"  {'':6s}  {nm:26s} {np.mean(v):8.1f} {np.mean(o):8.1f} "
                  f"{np.mean(dd):+8.1f}  [{', '.join(f'{x:+.1f}' for x in dd)}] "
                  f"{sum(1 for x in dd if x > 0)}/4")

    print("\n  참고: 리더보드는 HistGB 60+40 = 989, CatBoost 60+40 = 987 로 동점이었다.")
    print("  같은 총량에서 차이가 없으면 그 동점이 지금도 유지된다는 뜻이다.")
