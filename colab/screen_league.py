# -*- coding: utf-8 -*-
"""투수 리그 이동 이력을 편상관으로 거른다. 로컬 CPU.

왜 이 축인가
    44열에 **리그 이력이 없다**. game_type 은 그 행이 어느 리그인지만 말한다.
    직전에 어디서 던졌는지, 승강 직후인지, 1군 경험이 얼마인지는 어디에도 없다.

    어제 basic 문서의 파생변수 12개를 걸렀는데 전부 탈락했다. count_state,
    count_pressure, hand_matchup 은 편상관이 정확히 0.0000 이었다 — 44열의
    선형결합이라 새 정보가 아니었다. 이 축은 다르다.

    두 리그를 오간 투수가 453명이다(퓨처스 투수 633명 중 71.6%). 콜업 직후
    투수와 강등 직후 투수는 상태가 다를 수 있고, 적용 대상이 넓다.

as-of 규약
    row_id 순서가 as-of 기준과 일치하는 것을 확인했다 — 투수 792명 전원에서
    asof_pitcher_n 이 row_id 순으로 단조 증가한다. 그러니 row_id 로 정렬해
    **그 행 이전까지만** 누적하면 된다.

    추론에서는 학습 구간(2019~2024) 이력이 얼어붙는다. 2025 행은 그 투수의
    2024년 말 상태를 본다. 시즌 중 승강은 못 본다 — 규칙 4 때문이다.
    다른 평가 행을 참조하면 실격이다.

후보
    lg_f_share      지금까지 던진 공 중 퓨처스 비율
    lg_f_recent     최근 500구 중 퓨처스 비율
    lg_prev_f       직전 공이 퓨처스였나
    lg_is_switch    이 공이 직전과 다른 리그인가 (승강 직후 신호)
    lg_run          현재 리그에서 연속으로 던진 공 수 (log1p)
    lg_switches     지금까지 리그 전환 횟수 (log1p)
    lg_1gun_n       1군 누적 투구수 (log1p)
    lg_f_n          퓨처스 누적 투구수 (log1p)

판정
    44열로 통제한 뒤 |편상관| > 0.005 면 관문으로 보낸다. 어제와 같은 기준선이다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DATA = "open (1)/data"
VS = 2024
THRESH = 0.005
RECENT = 500

import features44 as F                                          # noqa: E402


def build(df):
    """row_id 순서로 투수별 누적. 전부 '그 행 이전까지' 다."""
    s = df.sort_values("row_id").reset_index(drop=True)
    f = (s["game_type"].astype(str) == "F").astype(np.float64)
    s["_f"] = f
    g = s.groupby("pitcher_id", sort=False)

    n_before = g.cumcount().to_numpy().astype(np.float64)
    f_before = (g["_f"].cumsum().to_numpy() - f.to_numpy())
    prev_f = g["_f"].shift(1).to_numpy()                 # 첫 공이면 NaN
    chg = np.where(np.isnan(prev_f), np.nan, (f.to_numpy() != prev_f).astype(float))

    # 전환 횟수. 첫 공의 NaN 은 0 으로 두고 누적한다.
    s["_chg"] = np.nan_to_num(chg, nan=0.0)
    sw_incl = g["_chg"].cumsum().to_numpy()
    sw_before = sw_incl - s["_chg"].to_numpy()

    # 현재 리그 연속 투구수. 전환 누적을 구간 id 로 써서 그 안에서 순번을 센다.
    s["_run"] = sw_incl
    run_pos = s.groupby(["pitcher_id", "_run"], sort=False).cumcount().to_numpy()

    # 최근 500구 중 퓨처스 비율. shift 로 자기 자신을 뺀 뒤 굴린다.
    sh = g["_f"].shift(1)
    roll = sh.groupby(s["pitcher_id"], sort=False).rolling(
        RECENT, min_periods=20).mean().reset_index(level=0, drop=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame({
            "row_id": s["row_id"].to_numpy(),
            "lg_f_share": np.where(n_before > 0, f_before / np.maximum(n_before, 1),
                                   np.nan),
            "lg_f_recent": roll.to_numpy(),
            "lg_prev_f": prev_f,
            "lg_is_switch": chg,
            "lg_run": np.log1p(run_pos.astype(np.float64)),
            "lg_switches": np.log1p(sw_before),
            "lg_1gun_n": np.log1p(np.maximum(n_before - f_before, 0)),
            "lg_f_n": np.log1p(np.maximum(f_before, 0)),
        })
    return out


def partial(x, y, C):
    rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
    ry = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    sx, sy = rx.std(), ry.std()
    return 0.0 if sx < 1e-12 or sy < 1e-12 else float((rx * ry).mean() / (sx * sy))


if __name__ == "__main__":
    t0 = time.time()
    raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                      usecols=["row_id", "season", "game_type", "pitcher_id"])
    cand = build(raw)
    print(f"  파생 생성 {time.time()-t0:.0f}s")

    d = F.build(DATA, VS=VS)
    X44 = d["X44"].astype(np.float64)
    y = d["y"].astype(np.float64)
    m_tr = d["m_tr"]

    # features44 의 행 순서에 맞춘다. 원본 train.csv 순서를 기준으로 삼는다.
    order = raw["row_id"].to_numpy()
    cand = cand.set_index("row_id").reindex(order)
    assert len(cand) == len(X44), "행 수가 안 맞는다"

    med = np.nanmedian(X44[m_tr], 0)
    Xf = np.where(np.isnan(X44), med, X44)
    var_ok = Xf[m_tr].std(0) > 1e-9
    C = np.c_[np.ones(m_tr.sum()), Xf[m_tr][:, var_ok]]
    yt = y[m_tr]

    isf = d["is_f"]
    print(f"  학습 {m_tr.sum():,}행로 통제 (44열 + 절편)   기준선 |r| > {THRESH}\n")
    print(f"  {'후보':14s} {'결측':>7s} {'단순상관':>10s} {'편상관':>10s} "
          f"{'퓨처스행 편상관':>15s}  판정")
    keep = []
    m_f = m_tr & isf
    Cf = np.c_[np.ones(m_f.sum()), Xf[m_f][:, var_ok]]
    for name in cand.columns:
        v = cand[name].to_numpy(np.float64)
        vm = np.where(np.isnan(v), np.nanmedian(v[m_tr]), v)
        raw_r = float(np.corrcoef(vm[m_tr], yt)[0, 1]) if vm[m_tr].std() > 1e-12 else 0.0
        pc = partial(vm[m_tr], yt, C)
        pcf = partial(vm[m_f], y[m_f], Cf)
        ok = abs(pc) > THRESH or abs(pcf) > THRESH
        if ok:
            keep.append(name)
        print(f"  {name:14s} {np.isnan(v[m_tr]).mean()*100:6.1f}% {raw_r:+10.4f} "
              f"{pc:+10.4f} {pcf:+15.4f}  {'통과' if ok else ''}")

    print(f"\n  통과 {len(keep)}개: {keep if keep else '없음'}   {time.time()-t0:.0f}s")
    print("  퓨처스 행만 따로 본 이유: 이 축이 통한다면 리그를 오간 투수가 많은")
    print("  퓨처스 구간에서 먼저 보일 것이기 때문이다. 전체에서 묻힐 수 있다.")
    if keep:
        cand[keep].to_csv(os.path.join(SC, "_dl", "league_feats.csv.gz"),
                          index=False, compression="gzip")
        print(f"  통과분 저장: colab/_dl/league_feats.csv.gz")
