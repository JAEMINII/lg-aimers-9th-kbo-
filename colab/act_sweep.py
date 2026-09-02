# -*- coding: utf-8 -*-
"""활성함수를 매끄러운 것으로 바꾼다. ReLU / Softplus / GELU.

왜 이 축인가 — 우리 설정에 특히 맞는 이유가 있다
    Stage1 이 **2에폭, 약 1,200 스텝**뿐이다. ReLU 는 음수 입력에서 기울기가
    정확히 0 이라 죽은 유닛이 생기는데, 보통은 수십 에폭 동안 되살아날 기회가
    있다. 우리에겐 그 기회가 없다. 한 번 죽으면 학습이 끝날 때까지 죽어 있다.
    softplus 의 기울기는 sigmoid(x) 라 0 이 되지 않는다.

    그리고 목표가 Brier 다. 딱딱한 분류가 아니라 **매끄러운 조건부확률면**을
    맞히는 문제라, 조각선형 근사기(ReLU 망)보다 매끄러운 쪽이 유리할 여지가 있다.

    다만 softplus 만 재면 축을 반만 보는 것이다. softplus(0)=0.693 이라 활성값이
    통째로 양수로 밀려 LayerNorm 과 상호작용이 있고, 실전에서 매끄러운 활성함수
    중 더 자주 이기는 건 GELU 쪽이다. 기전(죽은 유닛 제거)은 셋이 공유하므로
    같이 태우는 비용이 같다.

ReLU 가 박힌 곳 (확인함)
    num_module.activation       수치 임베딩 안
    backbone.blocks.{0,1,2}.1   본체 3개

    임베딩 축은 방금 linear_relu 로 닫았다(periodic -43.5, piecewise -17.2,
    둘 다 0/3). 그러니 '본체만' 과 '전부' 를 갈라야 어느 쪽 효과인지 안다.

네 팔 (배치 구성, 3시드 짝비교)
    relu      현행
    sp_bb     본체 3개만 Softplus. 임베딩은 linear_relu 유지
    sp_all    4곳 전부 Softplus
    gelu_bb   본체 3개만 GELU(tanh 근사)

배치 함정 — 미리 막는다
    추론은 numpy+pandas 전용이다.
        softplus  np.logaddexp(0, x)                        정확히 재현됨
        GELU      PyTorch 기본은 erf 기반인데 numpy 에 erf 가 없다.
                  그래서 **처음부터** approximate='tanh' 로 학습한다.
                  0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3))) 로 재현된다.
    이걸 나중에 발견하면 학습을 통째로 다시 해야 한다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
#    (이름, 새 활성함수 생성자, 임베딩까지 바꾸나)
ARMS = (("relu", None, False),
        ("sp_bb", lambda: nn.Softplus(), False),
        ("sp_all", lambda: nn.Softplus(), True),
        ("gelu_bb", lambda: nn.GELU(approximate="tanh"), False))

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


def swap(model, factory, include_emb):
    """ReLU 모듈을 갈아끼운다. 이름으로 임베딩/본체를 구분한다."""
    n = 0
    for parent_name, parent in model.named_modules():
        for cname, child in list(parent.named_children()):
            if not isinstance(child, nn.ReLU):
                continue
            full = f"{parent_name}.{cname}" if parent_name else cname
            is_emb = full.startswith("num_module")
            if is_emb and not include_emb:
                continue
            setattr(parent, cname, factory())
            n += 1
    return n


def one(nm, factory, inc_emb, seed, tr_idx, t_isf, w10, season, first=False):
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("regular", ~t_isf), ("futures", t_isf)):
        idx, w = tr_idx[sel], w10[sel]
        torch.manual_seed(seed)                # 모델 생성 전에 건다
        torch.cuda.manual_seed_all(seed)
        m = G.make_model()
        if factory is not None:
            k = swap(m, factory, inc_emb)
            if first and br == "all":
                G.log(f"    {nm}: ReLU {k}개 교체 "
                      f"(임베딩 {'포함' if inc_emb else '제외'})")
        G.train(m, idx, 2, LR1, w=w, seed=seed, tag=f"{nm} {br} s{seed} S1")
        s2 = idx[season[idx] == VS - 1]
        ps = G.stage2_params(m)
        for e in range(EP2[br]):
            G.train(m, s2, 1, 2e-4, params=ps, seed=seed + e,
                    tag=f"{nm} {br} s{seed} S2e{e+1}")
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
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
    pos = pd.Series(np.arange(len(tr)), index=tr["row_id"].to_numpy())
    Xf = Xs.to_numpy(dtype=np.float32)[
        pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()]
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]

    season, isf = G.season.astype(np.float64), G.is_f
    old = season <= OLD_F_MAX
    c4 = np.where(old & isf, 0.0,
                  np.where(old & ~isf, 1.0,
                           np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
    Xn, Xc, cards = G.prep(np.concatenate([Xf, c4], 1), G.m_tr,
                           ci + [Xf.shape[1]])
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)

    tr_idx = np.where(G.m_tr)[0]
    t_isf = isf[tr_idx]
    w10 = np.where(t_isf & old[tr_idx], OLD_W, 1.0).astype(np.float64)
    G.log(f"  수치 {Xn.shape[1]}열 범주 {len(cards)}열  학습 {len(tr_idx):,}  "
          f"Stage1 2에폭 = 약 {int(np.ceil(len(tr_idx)/2048))*2:,} 스텝")

    RES = {}
    for nm, fac, inc in ARMS:
        t0 = time.time()
        RES[nm] = [one(nm, fac, inc, sd, tr_idx, t_isf, w10, season,
                       first=(i == 0)) for i, sd in enumerate(SEEDS)]
        for i, sd in enumerate(SEEDS):
            np.save(os.path.join(OUT, f"act_{nm}_r_s{sd}.npy"), RES[nm][i][0])
            np.save(os.path.join(OUT, f"act_{nm}_f_s{sd}.npy"), RES[nm][i][1])
        G.log(f"  {nm:8s} 끝 {time.time()-t0:.0f}s")

    def full(x):
        return np.where(is_f, x[1], x[0])

    G.log("\n" + "=" * 84)
    G.log(f"  {'활성함수':10s} {'전체':>8s} {'1군행':>8s} {'퓨처스행':>9s}"
          f"   현행 대비 (짝차이, 부호일치)")
    G.log("=" * 84)
    base = RES["relu"]
    for nm, fac, inc in ARMS:
        r = RES[nm]
        v = [float(np.mean([sc(full(x)) for x in r])),
             float(np.mean([sc(x[0], R) for x in r])),
             float(np.mean([sc(x[1], is_f) for x in r]))]
        if nm == "relu":
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
    G.log("    sp_bb 와 sp_all 의 차이가 '임베딩 안의 ReLU' 효과다.")
    G.log("    sp_bb 와 gelu_bb 가 같은 방향이면 기전(죽은 유닛)이 맞는 것이고,")
    G.log("    한쪽만 이기면 그 함수의 특성이지 기전이 아니다 — 후자면 관문")
    G.log("    과적합을 의심한다.")
    G.log("    판정선 +10 & 3/3. 용량 스윕과 같은 부류라 관문을 그대로")
    G.log("    리더보드로 환산하지 않는다.")
