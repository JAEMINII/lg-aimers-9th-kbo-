# -*- coding: utf-8 -*-
"""basic_model_pipeline.md 의 파생변수를 편상관으로 거른다. 로컬 CPU, 3분.

왜 편상관부터인가
    관문에 바로 올리면 후보 하나에 40분이다. 편상관은 3분이고, 지난번 8개
    후보 중 1개만 통과시켰는데 그 1개만 관문에서도 확정됐다.
    44열이 이미 담은 정보를 다시 넣으면 0 이 나온다는 걸 다섯 번 봤다
    (트랙맨 5종, 상호작용 인코딩 3종).

후보 (basic 문서에서 우리 44열에 없는 것만)
    count_state    볼-스트라이크 결합 범주 12칸. 우리가 기각한 '저카디널리티
                   범주화' 는 balls/strikes 를 **따로** 넣은 것이라 이건 다르다.
                   이건 두 축의 상호작용이다.
    count_pressure balls + strikes
    is_two_strikes / is_three_balls / is_full_count
    inning_group   이닝 4구간
    score_diff_abs 점수차 절댓값
    score_state    trailing / tie / leading
    is_close_game  |점수차| <= 1
    has_risp       2루 또는 3루 주자
    li_log1p       log1p(max(li, 0))
    hand_matchup   투수핸드 x 타자핸드 4칸

판정
    통제 후 |편상관| 이 0.005 를 넘으면 관문으로 보낸다. 지난번 기준선과 같다.
    통제는 44열 전체다. 44열로 이미 설명되는 몫을 걷어내고 남는 것만 본다.

    선형 편상관이라 비선형 상호작용은 과소평가된다. 그래서 통과선을 낮게 둔다.
    count_state 처럼 애초에 상호작용인 후보는 표에서 따로 읽는다.
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

import features44 as F                                          # noqa: E402


def cands(df):
    b = df["balls_before"].to_numpy(np.float64)
    s = df["strikes_before"].to_numpy(np.float64)
    inn = df["inning"].to_numpy(np.float64)
    sd = df["score_diff_pitcher_team"].to_numpy(np.float64)
    li = df["li"].to_numpy(np.float64)
    ph = df["pitcher_hand"].to_numpy(np.float64)
    bh = df["batter_hand"].to_numpy(np.float64)
    r2 = df["runner_on_2b"].to_numpy(np.float64)
    r3 = df["runner_on_3b"].to_numpy(np.float64)
    out = {
        "count_state": b * 3.0 + s,                 # 12칸을 하나의 축으로
        "count_pressure": b + s,
        "is_two_strikes": (s >= 2).astype(np.float64),
        "is_three_balls": (b >= 3).astype(np.float64),
        "is_full_count": ((b >= 3) & (s >= 2)).astype(np.float64),
        "inning_group": np.clip(np.ceil(inn / 3.0), 1, 4),
        "score_diff_abs": np.abs(sd),
        "score_state": np.sign(sd),
        "is_close_game": (np.abs(sd) <= 1).astype(np.float64),
        "has_risp": ((r2 > 0) | (r3 > 0)).astype(np.float64),
        "li_log1p": np.log1p(np.clip(li, 0, None)),
        "hand_matchup": (ph - 1.0) * 2.0 + (bh - 1.0),
    }
    return out


def partial(x, y, C):
    """C 로 통제한 뒤 x 와 y 의 상관."""
    rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]
    ry = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    sx, sy = rx.std(), ry.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float((rx * ry).mean() / (sx * sy))


if __name__ == "__main__":
    t0 = time.time()
    d = F.build(DATA, VS=VS, return_frame=True)
    frame = d["frame"]
    X44 = d["X44"].astype(np.float64)
    y = d["y"].astype(np.float64)
    m_tr = d["m_tr"]
    # 44열에는 결측이 있다(이력이 없는 신인 등). lstsq 가 NaN 을 못 받으므로
    # 학습 구간 중앙값으로 채운다. G.prep 이 학습 때 하는 것과 같은 처리다.
    med = np.nanmedian(X44[m_tr], 0)
    Xf = np.where(np.isnan(X44), med, X44)
    # 상수열은 lstsq 를 불안정하게 만든다. 절편이 이미 있으므로 뺀다.
    var_ok = Xf[m_tr].std(0) > 1e-9
    C = np.c_[np.ones(m_tr.sum()), Xf[m_tr][:, var_ok]]
    yt = y[m_tr]
    print(f"  통제열 {int(var_ok.sum())}개 (상수열 {int((~var_ok).sum())}개 제외)")

    print(f"  학습 {m_tr.sum():,}행 로 통제 (44열 + 절편)   기준선 |r| > {THRESH}")
    print(f"  {'후보':16s} {'단순상관':>10s} {'편상관':>10s}  판정")
    keep = []
    for name, v in cands(frame).items():
        v = np.asarray(v, np.float64)[m_tr]
        raw = float(np.corrcoef(v, yt)[0, 1]) if v.std() > 1e-12 else 0.0
        pc = partial(v, yt, C)
        ok = abs(pc) > THRESH
        if ok:
            keep.append(name)
        print(f"  {name:16s} {raw:+10.4f} {pc:+10.4f}  {'통과' if ok else ''}")
    print(f"\n  통과 {len(keep)}개: {keep if keep else '없음'}   {time.time()-t0:.0f}s")
    print("  선형 편상관이라 비선형 상호작용은 과소평가된다.")
    print("  count_state 는 애초에 상호작용이라 통과 못 해도 관문에서 한 번 볼 만하다.")
