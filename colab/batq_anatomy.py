# -*- coding: utf-8 -*-
"""타자 수준별 오차 해부. 가중치를 건드리기 전에 편향이 있는지부터 잰다.

재민님 제안
    "출루율 좋고 뛰어난 타자한테 제구 안 되는 건 당연하니 가중치를 줄이자"

먼저 물어야 할 것
    가중치를 줄이는 게 도움이 되려면 그 구간이 (a) 2025 에 없거나 비중이 다르거나
    (b) 라벨이 잡음이거나 (c) **모델이 이미 잘 맞히는데 손해를 보고 있거나** 여야 한다.
    엘리트 타자는 2025 에도 그대로 있으니 (a)는 아니다. 남는 건 (c) 다 —
    즉 그 구간에서 모델이 **편향**돼 있는가.

    편향이 0 이면 모델이 이미 배운 것이고, 가중치를 줄이면 그 구간만 나빠진다.
    편향이 있으면 재민님 지적이 맞고, 다만 고칠 도구는 가중치가 아니라
    구간 보정이나 피처일 수 있다.

    오차 해부를 카운트/이닝/주자/투수경험/월로는 했는데 타자 수준으로는 처음이다.

측정
    보류 자료 = VS=2024 관문 예측 (sample_n 의 base 팔, 시드 3개 평균).
    2024 는 학습에 안 들어갔다.

    엘리트 대리값 — 데이터에 출루율이 없다. 가장 가까운 둘을 쓴다.
        asof_batter_success_rate  낮을수록 투수가 고전한 타자
        asof_batter_n             출전이 많을수록 주전
    둘 다 as-of 라 누출이 없고 44열에 이미 있다.

기회 크기도 같이 낸다
    구간마다 **최적 시프트를 따로** 줬을 때 총점이 얼마나 오르는지.
    이게 '구간 보정으로 얻을 수 있는 상한' 이다. 작으면 이 축은 접는다.
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

d = F.build(DATA, VS=VS, return_frame=True)
season = d["season"].astype(int)
y = d["y"].astype(np.float64)
raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                  usecols=["row_id", "asof_batter_success_rate", "asof_batter_n"])
same = bool((raw["row_id"].to_numpy() == d["frame"]["row_id"].to_numpy()).all())
print(f"  행 순서 일치 {same}   (다르면 정렬한다)")
if not same:
    raw = raw.set_index("row_id").reindex(d["frame"]["row_id"].to_numpy()
                                          ).reset_index()

gate = np.where(season == VS)[0]
yv = y[gate]
p = np.mean([np.load(os.path.join(DL, f"sn_{VS}_base_s{s}.npy"))
             for s in (42, 1, 777)], 0)
assert len(p) == len(gate), f"예측 {len(p)} vs 관문 {len(gate)}"
sc0, sh = F.best_shift(p, yv)
q = np.clip(F.shift(p, sh), 1e-6, 1 - 1e-6)
r = yv.mean()
den = r * (1 - r)
print(f"  관문 {VS}  {len(gate):,}행  실제 {r:.4f}  예측 {q.mean():.4f}  "
      f"점수 {sc0:.1f}\n")

BSR = raw["asof_batter_success_rate"].to_numpy(np.float64)[gate]
BN = raw["asof_batter_n"].to_numpy(np.float64)[gate]


def anatomy(name, v, k=10, lo_is_elite=True):
    ok = np.isfinite(v)
    ed = np.nanquantile(v[ok], np.linspace(0, 1, k + 1))
    ed[0] -= 1e-9
    ed[-1] += 1e-9
    b = np.digitize(v, ed) - 1
    b = np.clip(b, 0, k - 1)
    b[~ok] = -1
    print(f"  {name}")
    print(f"    {'구간':>10s} {'비중':>7s} {'실제':>8s} {'예측':>8s} "
          f"{'편향':>9s} {'구간점수':>9s}")
    gain = 0.0
    for i in range(-1, k):
        m = b == i
        if m.sum() < 200:
            continue
        a, e = yv[m].mean(), q[m].mean()
        s_i = 100000 * (1 - ((q[m] - yv[m]) ** 2).mean() / den)
        # 이 구간만 최적 시프트를 줬을 때 총 Brier 감소분
        _, sh_i = F.best_shift(q[m], yv[m])
        qi = np.clip(F.shift(q[m], sh_i), 1e-6, 1 - 1e-6)
        gain += (((q[m] - yv[m]) ** 2).sum() - ((qi - yv[m]) ** 2).sum())
        lab = "결측" if i < 0 else f"{ed[i]:.3f}~{ed[i+1]:.3f}"
        print(f"    {lab:>10s} {m.mean()*100:6.1f}% {a:8.4f} {e:8.4f} "
              f"{e-a:+9.4f} {s_i:9.1f}")
    g = 100000 * gain / len(yv) / den
    print(f"    구간별 최적 시프트를 다 줬을 때 총점 상승 상한  {g:+.1f}\n")
    return g


g1 = anatomy("asof_batter_success_rate 10분위 (낮을수록 강타자)", BSR)
g2 = anatomy("asof_batter_n 10분위 (높을수록 주전)", BN)

# 두 축을 합쳐 '엘리트' 를 직접 정의
el = np.isfinite(BSR) & np.isfinite(BN)
thr_s = np.nanquantile(BSR[el], 0.25)
thr_n = np.nanquantile(BN[el], 0.75)
grp = np.where(el & (BSR <= thr_s) & (BN >= thr_n), 2,
               np.where(el & (BSR >= np.nanquantile(BSR[el], 0.75))
                        & (BN >= thr_n), 0, 1))
print("  엘리트 정의: 상대 성공률 하위 25% AND 출전 상위 25%")
print(f"    {'집단':>14s} {'비중':>7s} {'실제':>8s} {'예측':>8s} {'편향':>9s} "
      f"{'구간점수':>9s}")
for i, nm in ((2, "엘리트"), (1, "그 외"), (0, "약타자·주전")):
    m = grp == i
    if m.sum() < 200:
        continue
    s_i = 100000 * (1 - ((q[m] - yv[m]) ** 2).mean() / den)
    print(f"    {nm:>14s} {m.mean()*100:6.1f}% {yv[m].mean():8.4f} "
          f"{q[m].mean():8.4f} {q[m].mean()-yv[m].mean():+9.4f} {s_i:9.1f}")
print(f"\n  판정 — 상한이 {max(g1, g2):+.1f}점이다.")
print("  편향이 0 근처면 모델이 이미 배운 것이고, 가중치를 줄이면")
print("  그 구간 예측만 나빠진다. 상한이 크면 구간 보정을 검토한다.")
