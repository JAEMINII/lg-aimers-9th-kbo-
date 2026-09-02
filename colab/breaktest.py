# -*- coding: utf-8 -*-
"""어느 해에 '관계' 가 끊겼는지 찾는다. 수준 이동이 아니라 판별력으로 본다. 로컬 CPU.

물음
    1군에도 2024 에 체제 전환이 있었나? 성공률은 50.31 -> 48.97 (-1.34%p) 인데
    그건 2019 부터 이어진 표류 범위 안이다(-2.26 / -1.41 / -0.91 / -0.06 / -1.34).
    볼카운트별 상대 패턴도 2023->2024 가 가장 안정적이었다.

    그래도 평균과 패턴은 간접 증거다. 직접 물어야 한다 —
    '작년까지로 배운 것이 올해도 통하나'.

설계
    창을 밀면서 조건을 똑같이 맞춘다. 3년 학습 -> 1년 앞 채점, 리그별로 따로.
        2019~2021 -> 2022
        2020~2022 -> 2023
        2021~2023 -> 2024
    연도 간격이 전부 1년이라 [[gate-needs-year-gap]] 문제도 균등하다.

    채점은 최적 시프트에서 한다. 수준 이동은 시프트가 흡수하므로 남는 것은
    판별력뿐이다. 관계가 끊긴 해에는 이 값이 무너져야 한다.

    퓨처스 2023 을 양성 대조로 쓴다. 거기서 안 잡히면 이 방법 자체를 못 믿는다.
"""
import os
import sys
import time

import numpy as np
from catboost import CatBoostClassifier

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DATA = "open (1)/data"
HP = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
          verbose=0, allow_writing_files=False, thread_count=6, random_seed=42)

import features44 as F                                          # noqa: E402

if __name__ == "__main__":
    # VS 는 피처 생성의 기준선일 뿐이라 가장 늦은 값으로 한 번만 만든다.
    d = F.build(DATA, VS=2025)
    X, y = d["X44"].astype(np.float64), d["y"].astype(int)
    season, isf = d["season"].astype(int), d["is_f"]

    print(f"  {'리그':6s} {'학습':12s} {'채점':>6s} {'학습행':>10s} {'채점행':>9s} "
          f"{'성공률':>7s} {'판별력':>8s} {'최적시프트':>10s}")
    for tag, m_lg in (("1군", ~isf), ("퓨처스", isf)):
        prev = None
        for vy in (2022, 2023, 2024):
            tr = m_lg & (season >= vy - 3) & (season <= vy - 1)
            va = m_lg & (season == vy)
            mdl = CatBoostClassifier(**HP)
            mdl.fit(X[tr], y[tr])
            p = mdl.predict_proba(X[va])[:, 1]
            yv = y[va].astype(np.float64)
            s, c = F.best_shift(p, yv)
            delta = "" if prev is None else f"  전년대비 {s-prev:+8.1f}"
            print(f"  {tag:6s} {vy-3}~{vy-1:4d}  {vy:6d} {tr.sum():10,d} "
                  f"{va.sum():9,d} {yv.mean()*100:6.2f}% {s:8.1f} {c:+10.4f}{delta}")
            prev = s
        print()

    print("  읽는 법")
    print("    판별력이 그 해에 무너지면 관계가 끊긴 것이다. 수준만 움직였다면")
    print("    최적 시프트가 커지고 판별력은 유지된다.")
    print("    퓨처스 2023 에서 크게 무너져야 이 방법을 믿을 수 있다(양성 대조).")
