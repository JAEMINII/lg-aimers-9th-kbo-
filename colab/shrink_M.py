# -*- coding: utf-8 -*-
"""수축 강도 M 을 추측하지 말고 잰다. 셀마다 다르다.

왜 M 이 중요한가
    조언은 M = 10~30 을 권한다. 그런데 우리 셀은 표본이 수백~수천이다.
    M=20 이면 표본 900 셀의 수축률이 900/920 = 0.978 로 **사실상 수축이 없다**.
    그러면 막겠다던 소표본 잡음이 그대로 들어온다.

경험적 베이즈의 정답
    관측된 셀 비율 p_hat 은 참값 p 에 이항 잡음이 얹힌 것이다.
        Var(관측 편차) = Var(참 편차) + E[p(1-p)/n]
    따라서
        투수간분산 = Var(관측 편차) - E[p(1-p)/n]
        M*         = 셀내분산 / 투수간분산 = p(1-p) / 투수간분산

    투수간분산이 작을수록(= 투수들이 서로 안 다를수록) M 이 커진다. 극단적으로
    투수간 차이가 없으면 M = 무한대가 되어 전부 무조건부 평균으로 수축한다.
    이게 맞는 답이다 — 없는 신호를 표본잡음으로 흉내내지 않는다.

주의
    Var(관측 편차) 를 그냥 재면 잡음이 포함돼 있어 M 이 과소평가된다.
    반드시 E[p(1-p)/n] 을 빼야 한다. 그게 이 스크립트가 하는 일이다.
    빼고 나서 음수가 나오면 **투수간 실질 차이가 0** 이라는 뜻이고, 그 축은
    피처로 만들 가치가 없다.
"""
import os
import sys

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DATA = "open (1)/data"
VS = 2024
MIN_N = 30          # 이보다 작은 셀은 추정에서 뺀다


def measure(df, who, cellname, cell, target):
    """M* 와 투수간 표준편차를 돌려준다."""
    d = pd.DataFrame({"w": df[who].to_numpy(), "c": cell, "y": target})
    ov = d.groupby("w")["y"].agg(["sum", "count"])
    ov["p"] = ov["sum"] / ov["count"]
    g = d.groupby(["w", "c"])["y"].agg(["sum", "count"])
    g = g[g["count"] >= MIN_N].copy()
    if len(g) < 100:
        return None
    g["p_hat"] = g["sum"] / g["count"]
    g["p_ov"] = ov["p"].reindex(g.index.get_level_values(0)).to_numpy()
    dev = (g["p_hat"] - g["p_ov"]).to_numpy()
    n = g["count"].to_numpy(np.float64)
    pv = g["p_ov"].to_numpy()
    noise = float(np.mean(pv * (1 - pv) / n))       # E[p(1-p)/n]
    obs = float(np.var(dev))
    true_var = obs - noise
    pbar = float(target.mean())
    if true_var <= 0:
        return dict(cells=len(g), n_med=float(np.median(n)), obs_sd=np.sqrt(obs),
                    noise_sd=np.sqrt(noise), true_sd=0.0, M=np.inf,
                    share=0.0)
    return dict(cells=len(g), n_med=float(np.median(n)),
                obs_sd=np.sqrt(obs), noise_sd=np.sqrt(noise),
                true_sd=np.sqrt(true_var),
                M=pbar * (1 - pbar) / true_var,
                share=true_var / obs)


if __name__ == "__main__":
    use = ["season", "pitcher_id", "batter_id", "balls_before",
           "strikes_before", "outs_before", "inning", "num_runners_on",
           "runner_on_2b", "runner_on_3b", "batter_hand", "li",
           "control_success"]
    d = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                    usecols=use)
    d = d[d.season < VS].reset_index(drop=True)      # 학습 구간에서만 추정
    y = d["control_success"].to_numpy(np.float64)
    print(f"  추정 표본 {len(d):,}행 (season < {VS})   전체 성공률 {y.mean():.4f}\n")

    ahead = np.where(d.strikes_before > d.balls_before, 2,
                     np.where(d.balls_before > d.strikes_before, 0, 1))
    CELLS = {
        "bhand": d["batter_hand"].to_numpy(),
        "strikes": d["strikes_before"].to_numpy(),
        "balls": d["balls_before"].to_numpy(),
        "ahead": ahead,
        "twostrike": (d.strikes_before == 2).to_numpy().astype(np.int8),
        "runner": (d.num_runners_on > 0).to_numpy().astype(np.int8),
        "risp": ((d.runner_on_2b > 0) | (d.runner_on_3b > 0)).astype(np.int8),
        "late": (d.inning >= 7).to_numpy().astype(np.int8),
        "outs": d["outs_before"].to_numpy(),
        "hi_li": (d["li"].to_numpy() > np.nanmedian(d["li"])).astype(np.int8),
    }

    print("=" * 92)
    print(f"  {'축':12s} {'셀수':>7s} {'셀표본중앙':>10s} {'관측편차SD':>10s} "
          f"{'잡음SD':>8s} {'참편차SD':>9s} {'신호비중':>8s} {'M*':>9s}")
    print("=" * 92)
    rows = []
    for nm, c in CELLS.items():
        r = measure(d, "pitcher_id", nm, c, y)
        if r is None:
            print(f"  {nm:12s} 셀 부족")
            continue
        rows.append((nm, r))
        M = "무한" if not np.isfinite(r["M"]) else f"{r['M']:9.0f}"
        print(f"  {nm:12s} {r['cells']:7d} {r['n_med']:10.0f} "
              f"{r['obs_sd']:10.4f} {r['noise_sd']:8.4f} {r['true_sd']:9.4f} "
              f"{r['share']*100:7.1f}% {M:>9s}")

    print("\n  '신호비중' = 관측된 편차 분산 중 진짜인 비율. 나머지는 표본잡음이다.")
    print("  M* 는 그 비율을 정확히 상쇄하는 수축 강도다.")
    print("  조언의 M=10~30 과 비교해 보라 — 표본 수백짜리 셀에서 M=20 은")
    print("  수축률 0.98 로 사실상 수축을 안 하는 것이고, 잡음을 그대로 통과시킨다.")
    if rows:
        best = max(rows, key=lambda x: x[1]["true_sd"])
        print(f"\n  참편차가 가장 큰 축: {best[0]}  참 SD {best[1]['true_sd']:.4f}"
              f"  (신호비중 {best[1]['share']*100:.0f}%)")
