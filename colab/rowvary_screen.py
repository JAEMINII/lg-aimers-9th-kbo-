# -*- coding: utf-8 -*-
"""행마다 값이 변하는 후보들을 편상관으로 미리 거른다. 학습 없이.

오늘 배운 제약
    트랙맨 5종, 상호작용 인코딩 3종이 전부 0 이었다. 이유가 같다 —
    44열이 이미 담은 정보를 다른 모양으로 다시 넣은 것이었다.
      트랙맨은 (투수,시즌) 상수라 pitcher_id 임베딩(793개)과 중복
      투수x카운트는 '투수 성공률 + 카운트' 의 합이라 편상관에서 90% 사라짐

    그래서 새 후보는 두 조건을 통과해야 한다.
      1. 같은 투수의 두 행에서 값이 달라야 한다
      2. 기존 44열을 통제한 뒤에도 정답과의 상관이 남아야 한다

    2번을 학습 없이 재는 게 이 스크립트다. 관문 연도(2024)에서 잰다.

후보
    부하      pn_cur 대비 시즌 경과. 등판이 몰린 투수인가
    폼 추세   prev1/prev3/prev5 는 있는데 그 '차이'(상승/하락)는 없다
    타자 비대칭  투수는 12개인데 타자는 2개뿐이다. 타자 인시즌을 더 쪼갠다
"""
import os

import numpy as np
import pandas as pd

D = "open (1)/data"
ALPHA = 50.0


def partial_corr(x, y, Z):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(1)
    x, y, Z = x[ok], y[ok], Z[ok]
    Z = np.c_[np.ones(len(Z)), Z]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1], int(ok.sum())


if __name__ == "__main__":
    tr = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig")
    d = tr[tr.season == 2024].copy().reset_index(drop=True)
    y = d.control_success.to_numpy(float)
    print(f"2024 {len(d):,}행  성공률 {y.mean():.4f}")

    # 인시즌 복원 (제출 코드와 같은 방식). 2024 를 평가 시즌으로 두고 2019~2023 을 뺀다
    prev = tr[tr.season < 2024]
    base_pn = prev.groupby("pitcher_id").size()
    base_ps = prev.groupby("pitcher_id")["control_success"].sum()
    base_bn = prev.groupby("batter_id").size()
    base_bs = prev.groupby("batter_id")["control_success"].sum()
    pn = d.asof_pitcher_n.to_numpy(float) - d.pitcher_id.map(base_pn).fillna(0).to_numpy()
    ps = ((d.asof_pitcher_success_rate.fillna(0) * d.asof_pitcher_n).round().to_numpy()
          - d.pitcher_id.map(base_ps).fillna(0).to_numpy())
    bn = d.asof_batter_n.to_numpy(float) - d.batter_id.map(base_bn).fillna(0).to_numpy()
    bs = ((d.asof_batter_success_rate.fillna(0) * d.asof_batter_n).round().to_numpy()
          - d.batter_id.map(base_bs).fillna(0).to_numpy())
    pn, ps, bn, bs = (np.clip(v, 0, None) for v in (pn, ps, bn, bs))
    prior = float(prev.control_success.mean())

    cand = {}
    # --- 부하: 시즌 경과 대비 투구 축적
    month = d.game_month.to_numpy(float)
    elapsed = np.clip(month - 2, 1, None)          # 3월 개막 기준 경과 개월
    cand["부하_월평균투구"] = pn / elapsed
    cand["부하_로그"] = np.log1p(pn) - np.log1p(elapsed)

    # --- 폼 추세: 최근 1경기 대비 5경기
    p1 = d.asof_pitcher_prev1_game_success_rate.to_numpy(float)
    p3 = d.asof_pitcher_prev3_game_success_rate.to_numpy(float)
    p5 = d.asof_pitcher_prev5_game_success_rate.to_numpy(float)
    cand["추세_1빼기5"] = p1 - p5
    cand["추세_3빼기5"] = p3 - p5
    cand["추세_1빼기시즌"] = p1 - (ps + ALPHA * prior) / (pn + ALPHA)
    m1 = d.asof_pitcher_prev1_game_middle_rate.to_numpy(float)
    m5 = d.asof_pitcher_prev5_game_middle_rate.to_numpy(float)
    cand["추세_중간1빼기5"] = m1 - m5

    # --- 타자 쪽 비대칭 보완
    cand["타자_인시즌성공률"] = (bs + ALPHA * prior) / (bn + ALPHA)
    cand["타자_인시즌표본"] = np.log1p(bn)
    cand["타자_커리어대비"] = ((bs + ALPHA * prior) / (bn + ALPHA)
                          - d.asof_batter_success_rate.to_numpy(float))
    cand["투타_성공률차"] = ((ps + ALPHA * prior) / (pn + ALPHA)
                        - (bs + ALPHA * prior) / (bn + ALPHA))

    # 통제 변수: 44열 중 강한 것들 + 카운트 더미
    ctrl_cols = ["asof_pitcher_success_rate", "asof_batter_success_rate",
                 "asof_pitcher_prev1_game_success_rate",
                 "asof_pitcher_prev3_game_success_rate",
                 "asof_pitcher_prev5_game_success_rate",
                 "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
                 "li", "home_win_expectancy"]
    cnt = pd.get_dummies(d.balls_before.astype(int).astype(str) + "-"
                         + d.strikes_before.astype(int).astype(str), drop_first=True)
    # 44열의 파생 7개를 빠짐없이 통제한다. 처음에 b_is_succ 를 빼먹어서
    # 타자 인시즌 후보들이 살아남은 것처럼 보였다 — 이미 있는 피처였다.
    p_is = (ps + ALPHA * prior) / (pn + ALPHA)
    b_is = (bs + ALPHA * prior) / (bn + ALPHA)
    cm = ((d.balls_before.astype(int) * 3 + d.strikes_before.astype(int)) * 4
          + d.pitcher_hand.astype(int) * 2 + d.batter_hand.astype(int)).to_numpy()
    cmd = pd.get_dummies(pd.Series(cm).astype(str), drop_first=True).to_numpy(float)
    Z = np.c_[d[ctrl_cols].to_numpy(float), cnt.to_numpy(float),
              p_is, b_is, pn, bn, cmd]
    print(f"통제 변수 {Z.shape[1]}개 "
          f"(44열의 강한 것 + 카운트 + p_is_succ + b_is_succ + pn_cur + bn_cur + 카운트x매치업)")

    print()
    print(f"  {'후보':22s} {'단순상관':>10s} {'편상관':>10s}   판정")
    print("  " + "-" * 60)
    for name, x in cand.items():
        x = np.asarray(x, dtype=float)
        ok = np.isfinite(x)
        r0 = np.corrcoef(x[ok], y[ok])[0, 1]
        r1, n = partial_corr(x, y, Z)
        # 기준: plat_dev 급(약 0.01)이면 볼 만하다. 0.005 미만은 버린다.
        verdict = "후보" if abs(r1) >= 0.010 else ("약함" if abs(r1) >= 0.005 else "버림")
        print(f"  {name:22s} {r0:+10.4f} {r1:+10.4f}   {verdict}")

    print()
    print("  참고: 기존 피처의 편상관 (같은 통제 아래)")
    for c in ("plat_dev_proxy",):
        pass
    for c in ("asof_pitcher_reverse_rate", "asof_batter_middle_rate",
              "asof_pitcher_fastball_rate"):
        x = d[c].to_numpy(float)
        r1, _ = partial_corr(x, y, Z)
        print(f"    {c:34s} {r1:+.4f}")
