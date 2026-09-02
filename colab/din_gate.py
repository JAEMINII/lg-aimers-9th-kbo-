# -*- coding: utf-8 -*-
"""DIN-lite 를 submit_41(LB 1083) 구성 위에서 판정선까지 잰다.

지금까지 나온 것
    VS=2024, 41식 기준선(0.70 friend-TabM route + 0.30 CatBoost 내부혼합)에서
    DIN 단독 905.3, w=0.20 에서 +10.7. 단독이 880 판정선을 넘은 첫 후보다.
    VS=2022 는 CatBoost 배열이 없어 TabM 단독을 기준선으로 썼다 -> 다시 잰다.
    VS=2023 은 기준선이 -1.1 로 무너져 있었다 -> 쓰지 않는다.
        2023 은 퓨처스 체제 전환년이다. <2023 학습이 새 체제를 본 적이 없어
        기준선이 무너지고, 낮게 미는 것이면 뭐든 이득으로 보인다.

이 스크립트가 채우는 것 — 판정선은 두 폴드 같은 부호 & 3/3
    시드 3개를 각각 학습해 **시드별 이득**을 낸다. 평균만 보면 3/3 을 못 센다.
    두 폴드(2022, 2024) 다 돌린다.

기준선 — 폴드마다 41 구성을 최대한 재현한다
    TabM      h2h_{vs}_friend_s42        지인 전처리 + c4 + 라우팅, 시드 42
    CatBoost  2024  0.70*신50(10시드) + 0.30*구44(10시드)   41의 내부혼합 그대로
              2022  cb50fixed_2022 (구44 없음 — 근사치임을 표시한다)
    혼합      0.30*CatBoost + 0.70*TabM

주의 — 이건 관문이지 리더보드 추정이 아니다. 조합 변경의 전이율은 0.08 이었다.
"""
import os
import sys
import time

import numpy as np
import torch

SC = os.path.dirname(os.path.abspath(__file__))
for p in (SC, "/root/aimers", os.path.dirname(SC)):
    if p not in sys.path:
        sys.path.insert(0, p)
DATA = os.environ.get("AIMERS_DATA", "/root/open (1)/data")
DL = os.environ.get("AIMERS_DL", "/root/aimers/_dl")
SEEDS = tuple(int(x) for x in os.environ.get("DG_SEEDS", "42,1,777").split(","))
FOLDS = tuple(int(x) for x in os.environ.get("DG_FOLDS", "2022,2024").split(","))
WS = (0.05, 0.10, 0.15, 0.20, 0.25)

import features44 as F                                          # noqa: E402
import ctr_zoo as Z                                             # noqa: E402


def baseline(vs, y):
    tab = np.load(f"{DL}/h2h_{vs}_friend_s42.npy").astype(np.float64)
    if vs == 2024:
        cb = (0.70 * np.load(f"{DL}/pcgpu2024_c12_cmh_10.npy")
              + 0.30 * np.load(f"{DL}/pcgpu2024_base44_10.npy")).astype(np.float64)
        tag = "41 정확 재현"
    else:
        cb = np.load(f"{DL}/cb50fixed_{vs}.npy").astype(np.float64)
        tag = "근사 (구44 없음)"
    assert len(tab) == len(cb) == len(y), (len(tab), len(cb), len(y))
    return 0.30 * cb + 0.70 * tab, tag


if __name__ == "__main__":
    print(f"  폴드 {FOLDS}  시드 {SEEDS}   판정선: 두 폴드 같은 부호 & 3/3\n")
    for vs in FOLDS:
        Z.VS = vs
        t0 = time.time()
        D = Z.build_inputs()
        season, isf, y = D["season"], D["isf"], D["y"].astype(np.float64)
        gate = np.where(season == vs)[0]
        yv, isf_g = y[gate], isf[gate]
        tr_idx = np.where(D["m_tr"])[0]
        old = season <= 2022
        w = np.where(isf & old, 0.1, 1.0)
        base, tag = baseline(vs, yv)
        allm = np.ones(len(yv), bool)
        sc = lambda p, m=allm: F.best_shift(p[m], yv[m])[0]
        b0 = sc(base)
        print(f"\n  VS={vs}  기준선 {b0:.1f}  ({tag})   "
              f"1군 {sc(base, ~isf_g):.1f}  퓨처스 {sc(base, isf_g):.1f}")

        P = []
        for sd in SEEDS:
            ps = []
            for br, sel in (("all", np.ones(len(tr_idx), bool)),
                            ("regular", ~isf[tr_idx]), ("futures", isf[tr_idx])):
                ps.append(Z.fit("DIN", D, tr_idx[sel], w, sd, gate))
            P.append(np.where(isf_g, 0.6 * ps[0] + 0.4 * ps[2],
                              0.6 * ps[0] + 0.4 * ps[1]))
            print(f"    seed {sd:>3d}  DIN 단독 {sc(P[-1]):7.1f}  "
                  f"상관 {np.corrcoef(base, P[-1])[0,1]:.4f}", flush=True)
        np.save(f"{DL}/dg_{vs}_DIN.npy", np.asarray(P))
        p = np.mean(P, 0)
        print(f"    시드평균  단독 {sc(p):7.1f}  1군 {sc(p, ~isf_g):7.1f}  "
              f"퓨처스 {sc(p, isf_g):7.1f}  상관 {np.corrcoef(base, p)[0,1]:.4f}")
        print(f"    {'가중':>6s} {'시드평균':>9s}   {'시드별 이득':>28s}   {'부호':>5s}")
        for wt in WS:
            dd = [sc((1 - wt) * base + wt * q) - b0 for q in P]
            mu = float(np.mean(dd))
            gm = sc((1 - wt) * base + wt * p) - b0
            print(f"    w={wt:.2f} {gm:+9.1f}   "
                  + " ".join(f"{v:+7.1f}" for v in dd)
                  + f"   {sum(1 for v in dd if v > 0)}/{len(dd)}")
        print(f"    ({time.time()-t0:.0f}s)")
    print("\n  조합 변경의 전이율은 0.08 이었다. 관문 +10 이면 리더보드 +1 쯤이다.")
