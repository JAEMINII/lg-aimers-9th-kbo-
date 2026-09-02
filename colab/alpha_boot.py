# -*- coding: utf-8 -*-
"""라우팅 alpha 선택이 검증 표본 노이즈를 넘는지 부트스트랩으로 본다.

branch_post 에서 나온 것 (VS=2024, 4시드 평균, 고정 시프트 -0.005)
```
    최적 aF=0.4  aR=0.0   842.4
    aF=0.4 aR=0.2         841.4
    aF=0.4 aR=0.4         840.1     <- 현행에 가장 가까움
    확률 혼합 최적 0.2      840.5
```
aR=0 이 최적이라는 건 '1군 행에는 regular 브랜치를 아예 안 쓰는 게 낫다' 는 뜻이다.
그게 사실이면 regular 브랜치를 통째로 뺄 수 있고 추론 패스가 절반 가까이 줄어
같은 시간에 시드를 두 배 넣을 수 있다. 그래서 확인이 필요하다.

부트스트랩이 재는 것과 못 재는 것
    잰다     검증 행 25만 개를 다시 뽑았을 때 순위가 유지되는가
    못 잰다  시드를 바꿨을 때, 다른 연도였을 때
    시드별 예측이 저장돼 있지 않아 시드 불확실성은 여기서 못 본다.
    한 폴드라 연도 불확실성도 못 본다. 그 둘이 더 큰 위험이라는 걸
    오늘 여러 번 확인했으므로, 여기 결과는 '최소한의 확인' 으로만 읽는다.
"""
import os
import sys

import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
OUT = "/workspace/aimers/out"
FIXED = -0.005
EPS = 1e-6
NBOOT = 200

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv


def lg(p):
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


if __name__ == "__main__":
    A = {b: np.load(os.path.join(OUT, f"bp_{b}.npy"))
         for b in ("all", "futures", "regular")}
    la, lf, lr = lg(A["all"]), lg(A["futures"]), lg(A["regular"])
    r = yv.mean()
    denom = r * (1 - r)

    def pred(af, ar):
        z = np.where(is_f, la + af * (lf - la), la + ar * (lr - la))
        p = 1 / (1 + np.exp(-z))
        return F.shift(p, FIXED)

    cands = {"aF0.4 aR0.0": (0.4, 0.0), "aF0.4 aR0.2": (0.4, 0.2),
             "aF0.4 aR0.4": (0.4, 0.4), "aF0.6 aR0.0": (0.6, 0.0),
             "aF0.4 aR1.0": (0.4, 1.0)}
    P = {k: pred(*v) for k, v in cands.items()}
    print(f"  전체 {len(yv):,}행   고정 시프트 {FIXED}")
    for k, p in P.items():
        print(f"  {k:12s} {100000*(1-((p-yv)**2).mean()/denom):8.1f}")

    rng = np.random.default_rng(0)
    n = len(yv)
    wins = {k: 0 for k in cands}
    diffs = {k: [] for k in cands if k != "aF0.4 aR0.0"}
    for _ in range(NBOOT):
        idx = rng.integers(0, n, n)
        yb = yv[idx]
        db = yb.mean() * (1 - yb.mean())
        sc = {k: 100000 * (1 - ((P[k][idx] - yb) ** 2).mean() / db) for k in cands}
        wins[max(sc, key=sc.get)] += 1
        for k in diffs:
            diffs[k].append(sc["aF0.4 aR0.0"] - sc[k])
    print(f"\n  부트스트랩 {NBOOT}회 — 각 후보가 1위한 횟수")
    for k, v in sorted(wins.items(), key=lambda t: -t[1]):
        print(f"    {k:12s} {v:4d}/{NBOOT}")
    print("\n  aR=0.0 대비 차이 (양수면 aR=0 이 낫다)")
    for k, v in diffs.items():
        v = np.array(v)
        print(f"    vs {k:12s} 평균 {v.mean():+6.2f}  "
              f"[{np.percentile(v,2.5):+6.2f}, {np.percentile(v,97.5):+6.2f}]  "
              f"양수 {(v>0).mean()*100:5.1f}%")
    print("\n  주의: 시드·연도 불확실성은 여기 안 들어간다.")
    print("  오늘 리더보드에서 뒤집힌 것들이 전부 그 두 축이었다.")
