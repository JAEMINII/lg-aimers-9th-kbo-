# -*- coding: utf-8 -*-
"""옛 시즌의 pitcher_id 만 가린다. 상황 정보는 남기고 정체성만 최근으로.

왜 이게 앞선 실패의 수술 버전인가
    오늘 두 개가 같은 이유로 졌다.
        시즌감쇠 d1.2~2.0    -10.8 ~ -23.9   0/3
        신뢰도가중 relw       -7.3            0/3
    둘 다 옛 행 / 저표본 행을 **통째로** 눌렀다. 그런데 그 행들도 카운트,
    주자상황, 이닝 같은 정보는 멀쩡히 갖고 있다. 같이 버린 게 손해였다.

    여기서는 그 행을 그대로 학습에 쓰되 **pitcher_id 만 미지(0)로 바꾼다.**
    상황 정보는 100% 남고 정체성만 최근 시즌에서 배운다.

기전
    임베딩 792칸이 지금 2019~2024 평균으로 학습된다. 예측 대상은 2025 투수의
    현재 폼이다. 노쇠, 구종 변화, 부상 복귀가 전부 뭉개져 들어간다.
    asof_pitcher_* 아홉 열은 as-of 라 항상 최신인데, 임베딩만 과거에 묶여 있다.

위험 두 가지 — 그래서 범위를 훑는다
    표본     한 시즌만 쓰면 투수당 학습 행이 약 1,540 -> 320 으로 준다.
    콜드     그 시즌에 안 던진 투수는 임베딩이 초기값에 가깝게 남는다.

관문에서의 대응
    배치는 학습 2019~2024 / 시험 2025 라 '마지막 학습 시즌 = 2024' 를 살린다.
    관문은 학습 2019~2023 / 검증 2024 이므로 '마지막 학습 시즌 = 2023' 이다.
    그래서 절대 연도가 아니라 **VS 로부터 몇 시즌**으로 적는다.

    마스킹은 학습 행에만 건다. 검증 행은 진짜 pitcher_id 를 쓴다 — 배치에서
    시험 행이 진짜 id 를 갖는 것과 같다.

구현
    G.prep 이 범주를 1..k 로 인코딩하고 미지값을 0 으로 둔다. 그 0 이
    '학습에서 못 본 투수' 코드다. 마스킹은 그 0 을 그대로 쓴다 — 새 코드를
    만들 필요가 없고, 추론에서 신인 투수가 이미 그 경로를 탄다.

팔
    base       현행
    keep1      VS-1 시즌만 살림                 (관문 2023 / 배치 2024)
    keep2      VS-1, VS-2                       표본 2배
    keep1_pb   VS-1 만 살리되 batter_id 도 같이  타자 정체성도 최근으로
    rand50     연도 무관 무작위 50% 마스킹       정규화 효과만 분리
                 -> keep1 이 이기는데 rand50 도 이기면 '최신성' 이 아니라
                    그냥 임베딩 과의존을 푼 것이다. 둘을 갈라야 한다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, "/workspace/aimers")
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
SEEDS = (42, 1, 777)
LR1 = 3e-3
OLD_F_MAX = 2022
OLD_W = 0.1
EP2 = {"all": 1, "regular": 1, "futures": 4}
#      (이름, 몇 시즌 살릴까, batter_id 도?, 무작위비율)
ARMS = (("base", None, False, None), ("keep1", 1, False, None),
        ("keep2", 2, False, None), ("keep1_pb", 1, True, None),
        ("rand50", None, False, 0.5))

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
R = ~is_f
FULL = np.ones(len(gate), bool)


def sc(p, m=None):
    m = FULL if m is None else m
    return F.best_shift(p[m], yv[m])[0]


def one(seed, tr_idx, t_isf, w, season):
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("regular", ~t_isf), ("futures", t_isf)):
        idx, ww = tr_idx[sel], w[sel]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model()
        G.train(m, idx, 2, LR1, w=ww, seed=seed, tag=f"{br} s{seed} S1")
        s2 = idx[season[idx] == VS - 1]
        ps = G.stage2_params(m)
        for e in range(EP2[br]):
            G.train(m, s2, 1, 2e-4, params=ps, seed=seed + e,
                    tag=f"{br} s{seed} S2e{e+1}")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    return (0.6 * P["all"] + 0.4 * P["regular"],
            0.6 * P["all"] + 0.4 * P["futures"])


if __name__ == "__main__":
    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr[tr.season < VS])
    Xs = PP.transform_features(tr, hist, train_mode=True)
    cols = list(Xs.columns)
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                         usecols=["row_id"])
    pos = pd.Series(np.arange(len(tr)), index=tr["row_id"].to_numpy())
    Xf = Xs.to_numpy(dtype=np.float32)[
        pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()]
    catnames = list(PP.TABM_CATEGORICAL_FEATURES)
    ci = [cols.index(c) for c in catnames]

    season, isf = G.season.astype(np.float64), G.is_f
    old = season <= OLD_F_MAX
    c4 = np.where(old & isf, 0.0,
                  np.where(old & ~isf, 1.0,
                           np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
    Xn, Xc0, cards = G.prep(np.concatenate([Xf, c4], 1), G.m_tr,
                            ci + [Xf.shape[1]])
    G.Xn, G.cards = Xn, cards
    G.XN = torch.from_numpy(Xn)

    p_col = catnames.index("pitcher_id")
    b_col = catnames.index("batter_id")
    tr_idx = np.where(G.m_tr)[0]
    t_isf = isf[tr_idx]
    w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
    G.log(f"  범주열 {len(cards)}개, pitcher_id 는 {p_col}번 "
          f"(카디널리티 {cards[p_col]}), batter_id 는 {b_col}번")
    G.log(f"  학습 시즌 {sorted(set(season[tr_idx].astype(int)))}   "
          f"VS={VS}\n")

    RES = {}
    for nm, keep, pb, rnd in ARMS:
        Xc = Xc0.copy()
        if keep is not None:
            # 학습 행 중 '마지막 keep 시즌' 밖은 정체성을 가린다
            lo = VS - keep
            msk = tr_idx[season[tr_idx] < lo]
            Xc[msk, p_col] = 0
            if pb:
                Xc[msk, b_col] = 0
            n_uniq = len(np.unique(Xc[tr_idx, p_col])) - 1
            G.log(f"  {nm:9s} {len(msk):,}행 가림 (시즌 <{int(lo)}), "
                  f"살아남은 투수 {n_uniq}명 / {cards[p_col]-1}")
        elif rnd is not None:
            r = np.random.default_rng(0)
            msk = tr_idx[r.random(len(tr_idx)) < rnd]
            Xc[msk, p_col] = 0
            G.log(f"  {nm:9s} {len(msk):,}행 무작위 가림 ({rnd:.0%})")
        else:
            G.log(f"  {nm:9s} 마스킹 없음")
        G.XC = torch.from_numpy(Xc)

        t0 = time.time()
        RES[nm] = [one(sd, tr_idx, t_isf, w10, season) for sd in SEEDS]
        for i, sd in enumerate(SEEDS):
            np.save(os.path.join(OUT, f"pm_{nm}_r_s{sd}.npy"), RES[nm][i][0])
            np.save(os.path.join(OUT, f"pm_{nm}_f_s{sd}.npy"), RES[nm][i][1])
        G.log(f"  {nm:9s} 끝 {time.time()-t0:.0f}s")

    def full(x):
        return np.where(is_f, x[1], x[0])

    G.log("\n" + "=" * 90)
    G.log(f"  {'팔':10s} {'전체':>8s} {'1군행':>8s} {'퓨처스행':>9s}"
          f"   현행 대비 (짝차이, 부호일치)")
    G.log("=" * 90)
    base = RES["base"]
    for nm, *_ in ARMS:
        r = RES[nm]
        v = [float(np.mean([sc(full(x)) for x in r])),
             float(np.mean([sc(x[0], R) for x in r])),
             float(np.mean([sc(x[1], is_f) for x in r]))]
        if nm == "base":
            G.log(f"  {nm:10s} {v[0]:8.1f} {v[1]:8.1f} {v[2]:9.1f}   <- 현행")
            continue
        tail = []
        for lab, fn, msk in (("전체", full, FULL),
                             ("1군", lambda x: x[0], R),
                             ("퓨처스", lambda x: x[1], is_f)):
            d = [sc(fn(a), msk) - sc(fn(b), msk) for a, b in zip(r, base)]
            mu = float(np.mean(d))
            se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
            tail.append(f"{lab} {mu:+6.1f}+-{se:4.1f} "
                        f"{sum(1 for x in d if x > 0)}/{len(d)}")
        G.log(f"  {nm:10s} {v[0]:8.1f} {v[1]:8.1f} {v[2]:9.1f}   "
              + "  ".join(tail))

    G.log("\n  읽는 법")
    G.log("    keep1/keep2 가 이기고 rand50 은 지면 '최신성' 이 원인이다.")
    G.log("    rand50 도 같이 이기면 최신성이 아니라 임베딩 과의존을 푼 것이고,")
    G.log("      그러면 연도를 자를 게 아니라 드롭아웃으로 다뤄야 한다.")
    G.log("    keep2 > keep1 이면 표본 부족이 최신성 이득을 갉아먹는 것이다.")
    G.log("    학습분포 축이라 시즌가중과 같은 부류다 — 관문 부호를 믿는다.")
