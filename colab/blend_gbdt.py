# -*- coding: utf-8 -*-
"""CatBoost / GBDT / TabM 3원 비중을 지금 배치 구성에서 훑는다. 로컬 CPU.

배치 구성이 뭐냐가 답을 바꾼다
    HistGB 는 2026-08-16 에 '현제출대비 +0.4' 로 기각됐다. 그때 배치는
    CatBoost + 0.2 MLP 였고 1군만 채점했다. 지금은 0.14 CatBoost + 0.86 TabM,
    전체 R+F 다. HistGB 와 TabM 의 상관은 그때 잰 적이 없다.

    TabM 은 **1시드**로 잰다. submit_18(1048)이 1시드다. 4시드 평균 위에서
    부품을 재면 그 부품이 주는 분산 감소가 이미 없어진 상태라 과소평가된다.
    branch_post 에서 그렇게 재고 regular 브랜치를 잘못 뺐다.

    비교는 최적 시프트(판별력)와 고정 시프트(배치에서 실제 받는 값) 둘 다 찍는다.
"""
import itertools
import os
import sys

import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DL = os.path.join(SC, "_dl")
FIXED = -0.005
EPS = 1e-6

import features44 as F                                          # noqa: E402


def lg(p):
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


if __name__ == "__main__":
    y = np.load(os.path.join(DL, "gate_y.npy"))
    isf = np.load(os.path.join(DL, "gate_isf.npy"))
    P = {"CatBoost": np.load(os.path.join(SC, "cb_gate.npy")).astype(np.float64)}
    for nm in ("HistGB", "LightGBM", "LGBM_DART"):
        f = os.path.join(DL, f"gb_{nm}.npy")
        if os.path.exists(f):
            P[nm] = np.load(f)
    seeds = [s for s in ("s42", "s1", "s777", "s2")
             if os.path.exists(os.path.join(DL, f"fb_base_{s}.npy"))]
    TM1 = {s: np.load(os.path.join(DL, f"fb_base_{s}.npy")) for s in seeds}
    TM4 = np.load(os.path.join(DL, "fb_base.npy"))

    def sc(p, m=None):
        m = slice(None) if m is None else m
        return F.best_shift(p[m], y[m])[0]

    def scf(p, m=None):
        m = slice(None) if m is None else m
        return F.bss(F.shift(p[m], FIXED), y[m])

    print(f"  관문 {len(y):,}행  퓨처스 {isf.mean()*100:.1f}%  "
          f"실제 성공률 {y.mean():.4f}\n")
    print(f"  {'모델':12s} {'단독':>8s} {'1군':>8s} {'퓨처스':>8s}")
    for nm, p in P.items():
        print(f"  {nm:12s} {sc(p):8.1f} {sc(p, ~isf):8.1f} {sc(p, isf):8.1f}")
    for s in seeds:
        print(f"  {'TabM ' + s:12s} {sc(TM1[s]):8.1f} {sc(TM1[s], ~isf):8.1f} "
              f"{sc(TM1[s], isf):8.1f}")
    print(f"  {'TabM 4시드':12s} {sc(TM4):8.1f} {sc(TM4, ~isf):8.1f} "
          f"{sc(TM4, isf):8.1f}")

    print("\n  상관 (이게 앙상블 값어치를 정한다)")
    names = list(P) + ["TabM(s42)"]
    allp = dict(P); allp["TabM(s42)"] = TM1["s42"]
    print("  " + " " * 12 + "".join(f"{n:>11s}" for n in names))
    for a in names:
        row = "".join(f"{np.corrcoef(allp[a], allp[b])[0,1]:11.4f}" for b in names)
        print(f"  {a:12s}{row}")

    print("\n  2원: CatBoost + TabM (현행) 에 비중 훑기 — TabM 1시드 평균")
    best2 = {}
    for wcb in np.arange(0.0, 0.51, 0.02):
        v = [scf(wcb * P["CatBoost"] + (1 - wcb) * TM1[s]) for s in seeds]
        o = [sc(wcb * P["CatBoost"] + (1 - wcb) * TM1[s]) for s in seeds]
        best2[round(float(wcb), 2)] = (float(np.mean(o)), float(np.mean(v)))
    kb = max(best2, key=lambda k: best2[k][1])
    print(f"    최적 CatBoost 비중 {kb:.2f}  최적시프트 {best2[kb][0]:.1f}  "
          f"고정 {best2[kb][1]:.1f}")
    print(f"    현행 0.14           최적시프트 {best2[0.14][0]:.1f}  "
          f"고정 {best2[0.14][1]:.1f}")

    print("\n  3원: w_cb x CatBoost + w_gb x GBDT + (1-w_cb-w_gb) x TabM")
    print("       TabM 1시드마다 계산해 평균. 현행(0.14/0.00) 대비 차이.")
    base = best2[0.14][1]
    baseo = best2[0.14][0]
    gbs = [n for n in P if n != "CatBoost"]
    for gb in gbs:
        rows = []
        for wcb in (0.00, 0.07, 0.14, 0.21):
            for wgb in (0.05, 0.10, 0.15, 0.20):
                if wcb + wgb > 0.5:
                    continue
                v, o = [], []
                for s in seeds:
                    q = (wcb * P["CatBoost"] + wgb * P[gb]
                         + (1 - wcb - wgb) * TM1[s])
                    v.append(scf(q)); o.append(sc(q))
                rows.append((float(np.mean(v)), float(np.mean(o)), wcb, wgb))
        rows.sort(reverse=True)
        print(f"\n    [{gb}]  상위 5개 (고정시프트 기준)")
        for v, o, wcb, wgb in rows[:5]:
            print(f"      cb {wcb:.2f}  {gb} {wgb:.2f}  TabM {1-wcb-wgb:.2f}   "
                  f"고정 {v:8.1f} ({v-base:+6.1f})   최적 {o:8.1f} ({o-baseo:+6.1f})")

    print("\n  시드별로도 본다 — 평균만 보면 어느 시드에서 뒤집히는지 못 본다")
    for gb in gbs:
        rows = []
        for wcb in (0.00, 0.07, 0.14):
            for wgb in (0.05, 0.10, 0.15):
                d = [scf(wcb * P["CatBoost"] + wgb * P[gb]
                         + (1 - wcb - wgb) * TM1[s]) - scf(0.14 * P["CatBoost"]
                                                           + 0.86 * TM1[s])
                     for s in seeds]
                rows.append((float(np.mean(d)), wcb, wgb, d))
        rows.sort(reverse=True)
        m, wcb, wgb, d = rows[0]
        print(f"    {gb:10s} 최선 cb{wcb:.2f}/{wgb:.2f}  "
              f"평균 {m:+6.1f}  시드별 [{', '.join(f'{x:+.1f}' for x in d)}]  "
              f"{sum(1 for x in d if x > 0)}/{len(d)}")

    print("\n  판정: 고정시프트 기준 +8 이상이고 시드 전부 양수여야 넣는다.")
    print("  2026-08-16 의 기각선(+8)을 그대로 쓴다. 그때와 다른 건 배치 구성뿐이다.")
