# -*- coding: utf-8 -*-
"""1군 2024 ABS 도입이 '관계' 를 바꿨는지 본다. 수준만 봐서는 안 보인다.

왜 다시 보나
    시즌별 성공률만 보면 1군은 매끄러운 추세다 (54.95 -> 48.97, 연 -1.2pp).
    2024 는 -1.34pp 로 2020 의 -2.26pp 보다도 작다. 그래서 '단절 없음' 으로
    읽었는데, 그건 **수준**만 본 것이다.

    ABS 는 심판 개인차를 없애고 존을 표준화한다. 리그 평균은 별로 안 움직여도
    측정의 성질이 바뀐다. 관계가 바뀌는 것이지 수준이 바뀌는 게 아니다.

무엇으로 잡나 — 세 가지 서명
    ① 투수간 분산
       심판 잡음이 빠지면 관측된 투수별 성공률의 분산에서 잡음 몫이 줄어든다.
       참 분산(관측 - 이항잡음)이 커지거나 작아지는 불연속을 본다.

    ② 연도간 전이 상관
       투수의 Y년 성공률이 Y+1년을 얼마나 예측하나. 측정 규칙이 바뀌는 해에는
       이 상관이 떨어진다. 2023->2024(1군 ABS)가 2022->2023 보다 낮은지 본다.
       퓨처스는 2022->2023 이 떨어져야 한다 (거기가 ABS 도입).

    ③ 피처-표적 관계
       asof 열들의 표적 상관이 연도별로 어떻게 움직이나. 관계가 바뀌면
       상관 구조가 그 해에 꺾인다.

읽는 법
    ②가 1군 2023->2024 에서 꺾이면 **관문은 그걸 볼 수 없다**. 관문은
    학습 2019~2023 / 검증 2024 라 새 체제 1군 학습 데이터가 0개다.
    그러면 2024 를 강조하는 모든 장치(시즌가중, Stage2)의 값어치를
    관문이 구조적으로 과소평가한다는 뜻이 된다.
"""
import os

import numpy as np
import pandas as pd

DATA = "open (1)/data"
MIN_N = 100


def between_var(g):
    """투수별 성공률의 참 분산 = 관측분산 - 이항잡음."""
    g = g[g["count"] >= MIN_N]
    if len(g) < 30:
        return None
    p = (g["sum"] / g["count"]).to_numpy()
    n = g["count"].to_numpy(np.float64)
    pb = float((g["sum"].sum()) / (g["count"].sum()))
    noise = float(np.mean(pb * (1 - pb) / n))
    obs = float(np.var(p))
    return dict(k=len(g), obs=obs, noise=noise, true=max(obs - noise, 0.0))


if __name__ == "__main__":
    d = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                    usecols=["season", "game_type", "pitcher_id",
                             "control_success", "asof_pitcher_success_rate",
                             "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
                             "asof_pitcher_strike_rate", "balls_before",
                             "strikes_before"])
    seasons = sorted(d.season.unique())

    print("\n① 투수별 성공률의 참 분산 (관측 - 이항잡음).  표본 100구 이상 투수")
    print(f"  {'':8s} " + " ".join(f"{s:>9d}" for s in seasons))
    for g, lab in (("R", "1군"), ("F", "퓨처스")):
        row = []
        for s in seasons:
            m = (d.season == s) & (d.game_type == g)
            gg = d[m].groupby("pitcher_id")["control_success"].agg(["sum", "count"])
            r = between_var(gg)
            row.append("      —" if r is None else f"{np.sqrt(r['true'])*100:8.2f}%")
        print(f"  {lab:8s} " + " ".join(f"{v:>9s}" for v in row))
    print("  (참 표준편차. 심판 잡음이 빠지면 측정이 깨끗해져 값이 움직인다)")

    print("\n② 연도간 전이 상관.  같은 투수의 Y년 성공률 vs Y+1년 성공률")
    print(f"  {'':8s} " + " ".join(f"{a}->{b%100:02d}".rjust(9)
                                   for a, b in zip(seasons[:-1], seasons[1:])))
    for g, lab in (("R", "1군"), ("F", "퓨처스")):
        row = []
        for a, b in zip(seasons[:-1], seasons[1:]):
            out = {}
            for s in (a, b):
                m = (d.season == s) & (d.game_type == g)
                gg = d[m].groupby("pitcher_id")["control_success"].agg(["sum", "count"])
                gg = gg[gg["count"] >= MIN_N]
                out[s] = (gg["sum"] / gg["count"])
            j = pd.concat([out[a].rename("a"), out[b].rename("b")], axis=1).dropna()
            row.append(f"{j['a'].corr(j['b']):8.3f}" if len(j) >= 30
                       else f"  n={len(j)}")
        print(f"  {lab:8s} " + " ".join(f"{v:>9s}" for v in row))
    print("  1군은 2023->24, 퓨처스는 2022->23 이 ABS 도입 경계다.")
    print("  그 칸이 이웃보다 낮으면 측정 규칙이 바뀐 것이다.")

    print("\n③ asof 피처와 표적의 상관 (연도별)")
    cols = ["asof_pitcher_success_rate", "asof_pitcher_middle_rate",
            "asof_pitcher_ball_rate", "asof_pitcher_strike_rate"]
    for g, lab in (("R", "1군"), ("F", "퓨처스")):
        print(f"  [{lab}]")
        for c in cols:
            row = []
            for s in seasons:
                m = (d.season == s) & (d.game_type == g)
                x = d.loc[m, c].to_numpy(np.float64)
                y = d.loc[m, "control_success"].to_numpy(np.float64)
                ok = ~np.isnan(x)
                row.append(f"{np.corrcoef(x[ok], y[ok])[0,1]:8.4f}"
                           if ok.sum() > 1000 else "       —")
            print(f"    {c:32s} " + " ".join(row))

    print("\n  ②가 1군 2023->24 에서 꺾이면 관문은 그걸 못 본다 —")
    print("  관문은 학습 2019~2023 이라 새 체제 1군 학습 데이터가 0개다.")
    print("  그러면 2024 를 강조하는 장치의 값어치를 구조적으로 과소평가한다.")
