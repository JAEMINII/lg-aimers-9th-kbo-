# -*- coding: utf-8 -*-
"""조건부 as-of 집계를 만들고 편상관으로 거른다. 로컬 CPU.

왜 이 축인가
    오차 해부 결과 우리 모델은 **보정이 거의 완벽**했다. 열 개 십분위 전부
    편향 0.006 이하, 구간별 절편을 고쳐도 반반 검증에서 회수액이 0 이하다.
    구조가 망가진 게 아니라 정보가 부족하다.

    보정이 완벽하면 점수 = 100000 x Var(잡아낸 신호) / r(1-r) 이다.
    1057 -> 1100 은 분산 +4.1%, 표준편차 +2.0% 다. 새 정보가 필요하다.

    안 쓰는 정보의 가장 큰 덩어리는 **조건부 집계**다. 주어진 asof_pitcher_*
    9개가 전부 무조건부다. "이 투수의 통산 성공률" 은 있는데 "이 투수의
    2스트라이크 성공률" 은 없다.

통계가 이걸 제약한다 — TabR 을 죽인 것과 같은 계산
    통산 3000구 투수의 3-0 카운트는 39구뿐이라 SE 0.080 이고 신호 폭(0.05)의
    1.6배다. 무보정 조건부 비율은 그냥 잡음이다. 그래서
        (1) 셀을 잘게 안 쪼갠다. 2~4개짜리만 쓴다.
        (2) 반드시 수축한다. 경험적 베이즈로 통산 비율 쪽으로 당긴다.
        (3) 값이 아니라 **편차**를 넣는다 — 통산 비율은 이미 44열에 있으므로
            조건부 비율을 그대로 넣으면 대부분 중복이다.

    feat = shrink(조건부 as-of 비율) - (무조건 as-of 비율)
    shrink = (성공수 + K x 통산비율) / (표본수 + K)

누출 방지
    row_id 가 시간순임을 확인했다 (모든 시즌에서 game_month 비감소).
    각 행에서 **그 행 이전** 투구만 쓴다 (cumsum 후 자기 자신 빼기).
    주어진 asof_* 열과 같은 규약이다.

판정
    44열(pitcher_id 포함)을 통제한 편상관. 통제 후에도 남아야 임베딩과 기존
    열이 못 담은 정보다. 기준선은 지난 스크린과 같게 |r| > 0.005 로 둔다
    (basic 후보 12개는 전부 <0.0025 였고, 통과한 lg_f_share 가 -0.0174).
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
K = 150.0            # 수축 강도. 조건부 표본이 K 일 때 반씩 섞인다.

import features44 as F                                          # noqa: E402


def asof_dev(df, y, who, cell, K=K):
    """(who, cell) 별 as-of 성공률을 통산 as-of 성공률로부터의 편차로.

    cell 이 None 이면 무조건부(통산)만 계산해서 돌려준다.
    """
    g1 = df.groupby(who, sort=False)
    n_all = g1.cumcount().to_numpy(np.float64)
    s_all = g1["_y"].cumsum().to_numpy(np.float64) - y
    with np.errstate(invalid="ignore", divide="ignore"):
        base = np.where(n_all > 0, s_all / np.maximum(n_all, 1), np.nan)
    if cell is None:
        return base, n_all
    g2 = df.groupby([who, cell], sort=False)
    n_c = g2.cumcount().to_numpy(np.float64)
    s_c = g2["_y"].cumsum().to_numpy(np.float64) - y
    b = np.where(np.isnan(base), float(y.mean()), base)
    shrunk = (s_c + K * b) / (n_c + K)
    dev = shrunk - b
    # 통산 이력이 아예 없으면 정보가 없다 -> 0
    return np.where(n_all > 0, dev, 0.0), n_c


def resid(x, C):
    return x - C @ np.linalg.lstsq(C, x, rcond=None)[0]


def partial(x, y, C):
    rx, ry = resid(x, C), resid(y, C)
    sx, sy = rx.std(), ry.std()
    return 0.0 if sx < 1e-12 or sy < 1e-12 else float((rx * ry).mean() / (sx * sy))


if __name__ == "__main__":
    t0 = time.time()
    use = ["row_id", "season", "game_type", "pitcher_id", "batter_id",
           "balls_before", "strikes_before", "outs_before", "inning",
           "num_runners_on", "runner_on_2b", "runner_on_3b", "batter_hand",
           "li", "asof_pitcher_n", "control_success"]
    d = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                    usecols=use)
    assert d["row_id"].is_monotonic_increasing, "row_id 가 정렬돼 있지 않다"
    y = d["control_success"].to_numpy(np.float64)
    d["_y"] = y
    print(f"  {len(d):,}행  {time.time()-t0:.0f}s")

    # ---- 조건 셀 정의. 전부 2~4개짜리다. 잘게 쪼개면 잡음이 이긴다.
    ahead = np.where(d.strikes_before > d.balls_before, 2,
                     np.where(d.balls_before > d.strikes_before, 0, 1))
    risp = ((d.runner_on_2b > 0) | (d.runner_on_3b > 0)).astype(np.int8)
    CELLS = {
        "strikes": d["strikes_before"].to_numpy(),
        "balls": d["balls_before"].to_numpy(),
        "ahead": ahead,
        "twostrike": (d.strikes_before == 2).to_numpy().astype(np.int8),
        "runner": (d.num_runners_on > 0).to_numpy().astype(np.int8),
        "risp": risp,
        "late": (d.inning >= 7).to_numpy().astype(np.int8),
        "outs": d["outs_before"].to_numpy(),
        "bhand": d["batter_hand"].to_numpy(),
        "hi_li": (d["li"].to_numpy() > np.nanmedian(d["li"])).astype(np.int8),
    }

    feats = {}
    for nm, c in CELLS.items():
        d["_c"] = c
        dev, n_c = asof_dev(d, y, "pitcher_id", "_c")
        feats[f"pdev_{nm}"] = dev
        if nm in ("strikes", "ahead", "runner"):     # 타자 쪽도 몇 개만
            dev_b, _ = asof_dev(d, y, "batter_id", "_c")
            feats[f"bdev_{nm}"] = dev_b
        print(f"  pdev_{nm:10s} 만듦  조건부표본 중앙값 {np.median(n_c):7.0f}  "
              f"편차 sd {dev.std():.5f}  {time.time()-t0:.0f}s")

    # ---- 편상관 스크린
    dd = F.build(DATA, VS=VS)
    X44 = dd["X44"].astype(np.float64)
    yy = dd["y"].astype(np.float64)
    m_tr, isf = dd["m_tr"], dd["is_f"]
    assert len(X44) == len(d)
    med = np.nanmedian(X44[m_tr], 0)
    Xf = np.where(np.isnan(X44), med, X44)
    ok = Xf[m_tr].std(0) > 1e-9
    an = d["asof_pitcher_n"].to_numpy()
    subs = {"전체": m_tr, "퓨처스": m_tr & isf, "저표본투수": m_tr & (an < 500),
            "고표본투수": m_tr & (an >= 3000)}
    C = {k: np.c_[np.ones(v.sum()), Xf[v][:, ok]] for k, v in subs.items()}

    print(f"\n  44열 통제 편상관   기준선 |r| > {THRESH}"
          f"   (통과 전례: lg_f_share -0.0174)\n")
    print(f"  {'후보':16s} " + " ".join(f"{k:>11s}" for k in subs) + "   판정")
    keep = []
    for nm, v in feats.items():
        row, best = [], 0.0
        for k, msk in subs.items():
            r = partial(v[msk], yy[msk], C[k])
            row.append(r)
            best = max(best, abs(r))
        good = best > THRESH
        if good:
            keep.append(nm)
        print(f"  {nm:16s} " + " ".join(f"{x:+11.4f}" for x in row)
              + f"   {'통과' if good else ''}")

    print(f"\n  통과 {len(keep)}개: {keep if keep else '없음'}   "
          f"{time.time()-t0:.0f}s")
    if keep:
        out = pd.DataFrame({k: feats[k] for k in keep})
        out.insert(0, "row_id", d["row_id"].to_numpy())
        out.to_csv(os.path.join(SC, "_dl", "cond_pass.csv.gz"), index=False,
                   compression="gzip")
        print("  저장: colab/_dl/cond_pass.csv.gz")
    print("\n  통과분은 한 번에 넣지 말고 가장 강한 것 하나부터 관문에 넣는다.")
    print("  5개 묶으면 3폴드 전부 음수, 최강 1개만 넣으면 전부 양수였던 전례가 있다.")
