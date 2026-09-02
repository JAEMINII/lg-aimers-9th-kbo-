# -*- coding: utf-8 -*-
"""진짜 출루율로 오차 해부를 다시 한다. 앞선 판은 대리값이었다.

앞선 판의 문제
    '타자 수준' 을 asof_batter_success_rate 하위 25% 로 잡았는데, 그 열은
    타격 능력이 아니라 '이 타자 상대로 투수가 의도한 제구를 성공한 비율' 이다.
    재민님이 말한 출루율과 다른 값이다.

이번엔 복원한 출루율을 쓴다
    리그 0.3484 (KBO 실제 0.35~0.37), 시즌별 0.3403~0.3577, 판정 99.8%.
    기존 타자 열과 상관 -0.0144 로 사실상 새 축이다.

묻는 것
    강타자 구간에서 모델이 편향돼 있는가. 있으면 재민님 지적이 맞다.
    없으면 모델이 이미 배운 것이고 가중치를 줄이면 그 구간만 나빠진다.
"""
import os
import sys

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
ROOT = os.path.dirname(SC)
DATA = os.path.join(ROOT, "open (1)", "data")
DL = os.path.join(SC, "_dl")
VS = 2024

import features44 as F                                          # noqa: E402

d = F.build(DATA, VS=VS)
season = d["season"].astype(int)
y = d["y"].astype(np.float64)
gate = np.where(season == VS)[0]
yv = y[gate]
p = np.mean([np.load(os.path.join(DL, f"sn_{VS}_base_s{s}.npy"))
             for s in (42, 1, 777)], 0)
sc0, sh = F.best_shift(p, yv)
q = np.clip(F.shift(p, sh), 1e-6, 1 - 1e-6)
r = yv.mean()
den = r * (1 - r)
obp = np.load(os.path.join(DL, "obp_asof.npy")).astype(np.float64)[gate]
print(f"  관문 {VS}  {len(gate):,}행  점수 {sc0:.1f}  "
      f"출루율 분포 {np.percentile(obp,[5,50,95]).round(3).tolist()}\n")

k = 10
ed = np.quantile(obp, np.linspace(0, 1, k + 1))
ed[0] -= 1e-9
ed[-1] += 1e-9
bb = np.clip(np.digitize(obp, ed) - 1, 0, k - 1)
print("  as-of 출루율 10분위 (위로 갈수록 강타자)")
print(f"    {'출루율':>13s} {'비중':>7s} {'실제':>8s} {'예측':>8s} "
      f"{'편향':>9s} {'구간점수':>9s}")
gain = 0.0
for i in range(k):
    m = bb == i
    a, e = yv[m].mean(), q[m].mean()
    s_i = 100000 * (1 - ((q[m] - yv[m]) ** 2).mean() / den)
    _, sh_i = F.best_shift(q[m], yv[m])
    qi = np.clip(F.shift(q[m], sh_i), 1e-6, 1 - 1e-6)
    gain += ((q[m] - yv[m]) ** 2).sum() - ((qi - yv[m]) ** 2).sum()
    print(f"    {ed[i]:.3f}~{ed[i+1]:.3f} {m.mean()*100:6.1f}% {a:8.4f} "
          f"{e:8.4f} {e-a:+9.4f} {s_i:9.1f}")
g = 100000 * gain / len(yv) / den
print(f"\n  구간별 최적 시프트를 다 줬을 때 총점 상승 상한  {g:+.1f}")

hi = obp >= np.quantile(obp, 0.9)
lo = obp <= np.quantile(obp, 0.1)
print(f"\n  상위 10% 강타자  실제 {yv[hi].mean():.4f}  예측 {q[hi].mean():.4f}  "
      f"편향 {q[hi].mean()-yv[hi].mean():+.4f}")
print(f"  하위 10% 약타자  실제 {yv[lo].mean():.4f}  예측 {q[lo].mean():.4f}  "
      f"편향 {q[lo].mean()-yv[lo].mean():+.4f}")
print(f"  두 집단 실제 성공률 차이 {yv[hi].mean()-yv[lo].mean():+.4f}")
print("\n  재민님 전제는 '강타자한테 제구가 안 된다' 였다.")
print("  위 차이가 음수면 맞고, 0 근처면 애초에 그런 효과가 없다.")
