# -*- coding: utf-8 -*-
"""트랙맨의 마지막 형태 — 투수 프로파일 x 그 행의 상황. 편상관으로 거른다.

왜 이 형태만 남았나
    트랙맨은 2019~2024 뿐이라 2025 행에 물리량을 직접 못 붙인다. 만들 수 있는 건
    과거 집계뿐이고 그건 (투수,시즌) 상수다. 그런데 44열에 pitcher_id 가
    793개짜리 학습 임베딩으로 들어가 있어 그런 값은 이미 표현된다.
    다섯 번 시도해서 전부 0 이었다.

        릴리스 흔들림 2열 +0.8   물리량 13열 +0.3   21열 -7.1
        2스트라이크 8열 -11.6    2스트라이크 변조 2열 +1.0

    빠져나갈 길은 '행마다 값이 변하게' 만드는 것뿐이다. 투수 프로파일을 그 행의
    카운트와 결합하면 같은 투수라도 카운트에 따라 값이 달라진다.
    오늘 확정한 등판 강도(+2.4)가 정확히 그 조건을 통과한 피처였다.

무엇을 만드나
    투구 단위로 붙인 121만 행에서 (투수, 카운트12) 별 물리 프로파일을 만든다.
        그 카운트에서의 평균 구속 / 상하무브
        그 투수 전체 평균 대비 편차 (카운트별 변조)
        구종 배합 (패스트볼 비율)
    행마다 그 행의 카운트에 해당하는 값을 찾아 붙인다.

    시간 규약: 시즌 Y 행에는 시즌 < Y 누적만. 2025 는 2019~2024 전부.

거를 기준
    44열 전부(파생 7개 포함)와 카운트 더미를 통제한 편상관.
    |r| >= 0.010 이면 관문으로 올리고 아니면 버린다.
    비교 기준: 기존 피처 asof_batter_middle_rate 가 같은 통제에서 -0.008,
    오늘 확정된 등판 강도가 +0.011 이었다.
"""
import os

import numpy as np
import pandas as pd

D = "open (1)/data"
ALPHA = 50.0
MIN_CELL = 30            # 셀 표본이 이보다 적으면 투수 전체 평균으로 후퇴


def partial_corr(x, y, Z):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(1)
    x, y, Z = x[ok], y[ok], Z[ok]
    Z = np.c_[np.ones(len(Z)), Z]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1], int(ok.sum())


if __name__ == "__main__":
    j = pd.read_csv("colab/pitch_join.csv.gz")
    tr = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig")
    key = tr[["row_id", "pitcher_id", "balls_before", "strikes_before"]]
    j = j.merge(key, on="row_id", how="left")
    j["cnt12"] = j.balls_before.astype(int) * 3 + j.strikes_before.astype(int)
    j["is_fb"] = (j.pitch_type_group.astype(str) == "fastball").astype(float)
    print(f"투구 단위 정합 {len(j):,}행")

    # (투수, 시즌, 카운트) 합계 -> 뒤에서 시즌을 누적해 '이전까지' 로 만든다
    VARS = [("rel_speed", "spd"), ("induced_vert_break", "ivb"), ("is_fb", "fb")]
    g = j.groupby(["pitcher_id", "season", "cnt12"])
    cell = pd.DataFrame({f"s_{t}": g[c].sum() for c, t in VARS})
    cell["n"] = g.size()
    g2 = j.groupby(["pitcher_id", "season"])
    tot = pd.DataFrame({f"s_{t}": g2[c].sum() for c, t in VARS})
    tot["n"] = g2.size()

    seasons = sorted(j.season.unique())
    rows = []
    cellD = {k: v for k, v in cell.groupby(level=[0, 1])}
    totD = {k: v for k, v in tot.groupby(level=[0, 1])}
    for pid in j.pitcher_id.dropna().unique():
        accC = {}          # cnt -> [s_spd, s_ivb, s_fb, n]
        accT = np.zeros(4)
        for s in seasons:
            for c in range(12):
                a = accC.get(c)
                rec = {"pitcher_id": pid, "season": s, "cnt12": c}
                if accT[3] >= 200:
                    tm = accT[:3] / accT[3]
                    if a is not None and a[3] >= MIN_CELL:
                        cm = a[:3] / a[3]
                    else:
                        cm = tm                 # 셀이 얇으면 투수 전체로 후퇴
                    rec["tmc_spd"] = cm[0] - tm[0]
                    rec["tmc_ivb"] = cm[1] - tm[1]
                    rec["tmc_fb"] = cm[2] - tm[2]
                else:
                    rec["tmc_spd"] = rec["tmc_ivb"] = rec["tmc_fb"] = np.nan
                rows.append(rec)
            if (pid, s) in cellD:
                for (_, _, c), r in cellD[(pid, s)].iterrows():
                    v = np.array([r.s_spd, r.s_ivb, r.s_fb, r.n], float)
                    accC[c] = v if c not in accC else accC[c] + v
            if (pid, s) in totD:
                r = totD[(pid, s)].iloc[0]
                accT = accT + np.array([r.s_spd, r.s_ivb, r.s_fb, r.n], float)
    prof = pd.DataFrame(rows)
    print(f"프로파일 {prof.shape}   유효 {prof.tmc_spd.notna().mean()*100:.1f}%")
    prof.to_csv("colab/tm_cnt_profile.csv", index=False)

    # ---- 2024 에서 편상관
    d = tr[tr.season == 2024].copy().reset_index(drop=True)
    d["cnt12"] = d.balls_before.astype(int) * 3 + d.strikes_before.astype(int)
    d = d.merge(prof, on=["pitcher_id", "season", "cnt12"], how="left")
    y = d.control_success.to_numpy(float)
    print(f"2024 커버리지 {d.tmc_spd.notna().mean()*100:.1f}%")

    prev = tr[tr.season < 2024]
    bpn = prev.groupby("pitcher_id").size()
    bps = prev.groupby("pitcher_id")["control_success"].sum()
    bbn = prev.groupby("batter_id").size()
    bbs = prev.groupby("batter_id")["control_success"].sum()
    pn = np.clip(d.asof_pitcher_n - d.pitcher_id.map(bpn).fillna(0), 0, None).to_numpy(float)
    ps = np.clip((d.asof_pitcher_success_rate.fillna(0) * d.asof_pitcher_n).round()
                 - d.pitcher_id.map(bps).fillna(0), 0, None).to_numpy(float)
    bn = np.clip(d.asof_batter_n - d.batter_id.map(bbn).fillna(0), 0, None).to_numpy(float)
    bs = np.clip((d.asof_batter_success_rate.fillna(0) * d.asof_batter_n).round()
                 - d.batter_id.map(bbs).fillna(0), 0, None).to_numpy(float)
    pr = float(prev.control_success.mean())
    ctrl = ["asof_pitcher_success_rate", "asof_batter_success_rate",
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate",
            "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
            "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
            "li", "home_win_expectancy"]
    cntd = pd.get_dummies(d.cnt12.astype(str), drop_first=True).to_numpy(float)
    el = np.clip(d.game_month.to_numpy(float) - 2.0, 1.0, None)
    Z = np.c_[d[ctrl].to_numpy(float), cntd,
              (ps + ALPHA * pr) / (pn + ALPHA), (bs + ALPHA * pr) / (bn + ALPHA),
              pn, bn, np.log1p(pn) - np.log1p(el)]     # 등판 강도까지 통제
    print(f"통제 {Z.shape[1]}개 (44열 강한 것 + 카운트12 + 파생 + 등판강도)")

    print()
    print(f"  {'후보':16s} {'단순상관':>10s} {'편상관':>10s}   판정")
    print("  " + "-" * 54)
    for c in ("tmc_spd", "tmc_ivb", "tmc_fb"):
        x = d[c].to_numpy(float)
        ok = np.isfinite(x)
        r0 = np.corrcoef(x[ok], y[ok])[0, 1]
        r1, n = partial_corr(x, y, Z)
        v = "후보" if abs(r1) >= 0.010 else ("약함" if abs(r1) >= 0.005 else "버림")
        print(f"  {c:16s} {r0:+10.4f} {r1:+10.4f}   {v}   n={n:,}")
    print()
    print("  기준: 등판 강도(오늘 확정)가 +0.0111, asof_batter_middle_rate 가 -0.008")
