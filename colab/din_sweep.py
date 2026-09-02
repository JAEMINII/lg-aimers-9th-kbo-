# -*- coding: utf-8 -*-
"""DIN 의 세 설계축을 스윕한다 — 기억 길이 K, query 축, 기억 내용.

왜 이 셋인가
    K       배치 시퀀스 표의 평균길이가 15.9/16 이다. **포화 상태**라 임의로 잘린다.
    query   지금은 카운트12 x 타자손이다. 오늘 1군 잔여분석 1위가 카운트(상한 14.2),
            2위가 주자상황(7.4)이었다. 축을 바꾸면 다른 잔여를 먹는다.
    기억    지금은 성공/실패 2값이다. 복원 라벨은 5범주(성공/reverse/middle/ball/strike)
            이고 99.94% 복원된다. 더 풍부한 기억이 된다.

핵심 가설
    오늘 네 구조(DIN/SENet/BST/DCNv2)와 MNCA 가 전부 +8~+10 한 자리에 모였다.
    즉 이득은 아키텍처가 아니라 **상관 낮은 구성원 한 자리**에서 온다.
    그렇다면 서로 다른 query 축의 DIN 여럿은 **서로도 상관이 낮아야** 하고,
    그러면 한 자리가 아니라 여러 자리를 채울 수 있다. 변종 간 상관을 같이 낸다.

판정선은 그대로 — 두 폴드 같은 부호 & 3/3.
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
for p in (SC, "/root/aimers", os.path.dirname(SC)):
    if p not in sys.path:
        sys.path.insert(0, p)
DL = os.environ.get("AIMERS_DL", "/root/aimers/_dl")
SEEDS = tuple(int(x) for x in os.environ.get("DS_SEEDS", "42,1,777").split(","))
FOLDS = tuple(int(x) for x in os.environ.get("DS_FOLDS", "2022,2024").split(","))

import features44 as F                                          # noqa: E402
import ctr_zoo as Z                                             # noqa: E402
from din_gate import baseline                                   # noqa: E402

# (이름, K, query축, 기억 5범주 여부)
ARMS = [
    ("base_K16",   16, "cnt_hand", False),   # 현행 = 이미 통과한 구성
    ("K64",        64, "cnt_hand", False),
    ("q_base",     16, "base",     False),
    ("q_inn",      16, "inn",      False),
    ("mem5",       16, "cnt_hand", True),
    ("K64_mem5",   64, "cnt_hand", True),
]


def query_col(o, axis):
    b, s = o.balls_before.astype(int), o.strikes_before.astype(int)
    if axis == "cnt_hand":
        return np.clip(b * 3 + s, 0, 11) * 2 + o.batter_hand.astype(int)
    if axis == "base":
        return pd.factorize(o.base_state.astype(str))[0] * 3 + o.outs_before.astype(int)
    if axis == "inn":
        return np.clip(o.inning.fillna(1).astype(int), 1, 9) * 2 + o.batter_hand.astype(int)
    raise ValueError(axis)


if __name__ == "__main__":
    print(f"  폴드 {FOLDS}  시드 {SEEDS}   판정선 두 폴드 & 3/3\n")
    keep = {}
    for vs in FOLDS:
        Z.VS = vs
        base_D = None
        for nm, K, ax, mem5 in ARMS:
            Z.KSEQ = K
            Z._QAX, Z._MEM5 = ax, mem5
            t0 = time.time()
            D = Z.build_inputs()
            season, isf, y = D["season"], D["isf"], D["y"].astype(np.float64)
            gate = np.where(season == vs)[0]
            yv, isf_g = y[gate], isf[gate]
            tr = np.where(D["m_tr"])[0]
            w = np.where(isf & (season <= 2022), 0.1, 1.0)
            base, tag = baseline(vs, yv)
            allm = np.ones(len(yv), bool)
            sc = lambda p, m=allm: F.best_shift(p[m], yv[m])[0]
            b0 = sc(base)
            P = []
            for sd in SEEDS:
                ps = [Z.fit("DIN", D, tr[sel], w, sd, gate)
                      for sel in (np.ones(len(tr), bool), ~isf[tr], isf[tr])]
                P.append(np.where(isf_g, 0.6*ps[0] + 0.4*ps[2],
                                  0.6*ps[0] + 0.4*ps[1]))
            p = np.mean(P, 0)
            keep[(vs, nm)] = p
            np.save(f"{DL}/ds_{vs}_{nm}.npy", np.asarray(P))
            dd = [sc(0.85*base + 0.15*q) - b0 for q in P]
            mu = float(np.mean(dd))
            print(f"  VS={vs} {nm:10s} 단독 {sc(p):8.1f} 상관 "
                  f"{np.corrcoef(base, p)[0,1]:.4f}  w=0.15 {mu:+6.1f}  "
                  + " ".join(f"{v:+6.1f}" for v in dd)
                  + f"  {sum(1 for v in dd if v>0)}/3  ({time.time()-t0:.0f}s)",
                  flush=True)
        # 변종끼리 얼마나 다른가 — 여러 자리를 채울 수 있는지가 여기 달렸다
        names = [a[0] for a in ARMS]
        print(f"\n  VS={vs} 변종 간 상관")
        print("    " + " " * 11 + "".join(f"{n[:9]:>10s}" for n in names))
        for a in names:
            print(f"    {a:11s}" + "".join(
                f"{np.corrcoef(keep[(vs,a)], keep[(vs,b)])[0,1]:10.3f}" for b in names))
        print(flush=True)
