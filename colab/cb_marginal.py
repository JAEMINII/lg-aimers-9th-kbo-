# -*- coding: utf-8 -*-
"""TabM 이 **이미 고쳐진 상태**에서 CatBoost 를 추가로 고치면 얼마인가.

왜 다시 재나
    앞선 측정에서 혼합 차이가 +10.4 로 단독 차이(+6.7)보다 컸다. 원인은 내가
    쓴 TabM 기준선(sc_s42.npy)이 **아직 누출된 판본**이어서다. 그 +10.4 는
    'CatBoost 를 고친 이득' 이 아니라 '혼합의 절반을 고친 이득' 이다.

    배치(submit_30, LB 1069)에서 TabM 은 이미 as-of 다. 그 위에서 CatBoost 를
    추가로 고쳤을 때의 **한계 이득**이 실제로 얻을 값이다.

재료 (전부 VS=2024 관문에서 나온 것)
    TabM  pf_2024_asof_s{42,1,777}.npy    platfix 실험의 as-of 팔
          pf_2024_friend_s{...}.npy       같은 실험의 누출 팔 (대조용)
    CB    cbpf_{asof,leaky}_s{...}.npy    방금 만든 것
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/aimers")
import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402

OUT = "/workspace/aimers/out"
is_f, yv = G.is_f[G.gate], G.yv
R = ~is_f
SEEDS_T = (42, 1, 777)
SEEDS_C = (42, 1234, 2025)


def sc(p, m=None):
    m = np.ones(len(yv), bool) if m is None else m
    return F.best_shift(p[m], yv[m])[0]


def load(pat, seeds):
    out = []
    for s in seeds:
        p = os.path.join(OUT, pat.format(s))
        if os.path.exists(p):
            out.append(np.load(p))
    return out


TM = {k: load(f"pf_2024_{k}_s{{}}.npy", SEEDS_T) for k in ("asof", "friend")}
CB = {k: load(f"cbpf_{k}_s{{}}.npy", SEEDS_C) for k in ("asof", "leaky")}
for k in TM:
    print(f"  TabM {k:7s} {len(TM[k])}개")
for k in CB:
    print(f"  CB   {k:7s} {len(CB[k])}개")
if not all(TM.values()) or not all(CB.values()):
    raise SystemExit("예측 파일이 모자란다")

tm_a, tm_f = np.mean(TM["asof"], 0), np.mean(TM["friend"], 0)
cb_a, cb_f = np.mean(CB["asof"], 0), np.mean(CB["leaky"], 0)

print(f"\n  단독   TabM asof {sc(tm_a):7.1f}  leaky {sc(tm_f):7.1f}")
print(f"         CB   asof {sc(cb_a):7.1f}  leaky {sc(cb_f):7.1f}")

print(f"\n  혼합 (CB 0.30 / TabM 0.70)")
print(f"  {'TabM':>8s} {'CatBoost':>10s} {'전체':>9s} {'1군':>9s} {'퓨처스':>9s}")
grid = {}
for tn, tp in (("leaky", tm_f), ("asof", tm_a)):
    for cn, cp in (("leaky", cb_f), ("asof", cb_a)):
        p = 0.30 * cp + 0.70 * tp
        grid[(tn, cn)] = p
        print(f"  {tn:>8s} {cn:>10s} {sc(p):9.1f} {sc(p, R):9.1f} "
              f"{sc(p, is_f):9.1f}")

base = sc(grid[("leaky", "leaky")])
print(f"\n  둘 다 누출 (배치 이전 상태)            {base:9.1f}")
print(f"  TabM 만 고침 (submit_30, LB 1069)     "
      f"{sc(grid[('asof','leaky')]):9.1f}  {sc(grid[('asof','leaky')])-base:+8.1f}")
print(f"  CB 만 고침                            "
      f"{sc(grid[('leaky','asof')]):9.1f}  {sc(grid[('leaky','asof')])-base:+8.1f}")
print(f"  둘 다 고침                            "
      f"{sc(grid[('asof','asof')]):9.1f}  {sc(grid[('asof','asof')])-base:+8.1f}")
marg = sc(grid[("asof", "asof")]) - sc(grid[("asof", "leaky")])
print(f"\n  **한계 이득** — TabM 이 이미 고쳐진 위에서 CB 를 더 고치면 "
      f"{marg:+.1f}")

# 시드 짝차이로 오차막대
dd = []
for i in range(min(len(CB["asof"]), len(CB["leaky"]))):
    a = 0.30 * CB["asof"][i] + 0.70 * tm_a
    b = 0.30 * CB["leaky"][i] + 0.70 * tm_a
    dd.append(sc(a) - sc(b))
mu = float(np.mean(dd))
se = float(np.std(dd, ddof=1)) / np.sqrt(len(dd))
print(f"  시드 짝차이 {mu:+.1f} +- {se:.1f}  t={mu/max(se,1e-9):.2f}  "
      f"{sum(1 for x in dd if x>0)}/{len(dd)}   "
      f"[{', '.join(f'{v:+.1f}' for v in dd)}]")
print("\n  이 한계 이득이 submit_30 위에 얹을 때 기대할 값이다.")
print("  전이율은 결함 수정 부류라 0.5 쯤으로 본다 (plat_dev TabM 이 0.5 였다).")
