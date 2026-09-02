# -*- coding: utf-8 -*-
"""TabM 을 '우리 전처리' 와 '지인 전처리' 로 각각 학습해 맞대본다.

왜
    우리가 실은 submit_14(1041)는 지인 전처리를 쓴다. 그런데 관문 실험은 전부
    우리 전처리로 돌렸다. 두 축이 어긋난 채로 왔고, TabM 에 대해 둘을 직접
    비교한 적이 없다.

    두 구현은 이름·순서가 같지만 값이 다르다(plat_dev 최대 2.5e-2). 그리고
    flat MLP 에서는 우리 쪽이 나았다 — 우리 937.2 / 지인 927.7.
    TabM 도 그렇다면 submit_14 는 잘못된 전처리로 만들어진 것이다.

설정은 submit_14 와 동일: ep2 Stage1 + 마지막 시즌 Stage2 1epoch, k=32,
all/futures/regular 3브랜치 0.6:0.4. 시드 3개.
채점은 전체(R+F). 단일 시드는 24점씩 흔들려 판정에 못 쓴다.
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

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)
VS = int(os.environ.get("VS", 2024))


def friend_matrix():
    """지인 경로로 44피처. 룩업은 학습 구간에서만 만든다."""
    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr[tr.season < VS])
    X = PP.transform_features(tr, hist, train_mode=True)
    # G 의 행 순서(features44 기준)와 맞춘다. 둘 다 row_id 로 정렬해 대응시킨다.
    order = tr["row_id"].to_numpy()
    return X.to_numpy(dtype=np.float32), order, list(X.columns)


def run(tag, X, cat_idx, seed):
    Xn, Xc, cards = G.prep(X, G.m_tr, cat_idx)
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    P = {}
    tr_idx = np.where(G.m_tr)[0]
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("futures", G.is_f[tr_idx]), ("regular", ~G.is_f[tr_idx])):
        idx = tr_idx[sel]
        torch.manual_seed(seed)                 # 초기화까지 덮는다
        torch.cuda.manual_seed_all(seed)
        m = G.make_model()
        G.train(m, idx, 2, 2e-3, seed=seed, tag=f"{tag} {br} S1")
        s2 = G.season[idx] == VS - 1
        G.train(m, idx[s2], 1, 2e-4, params=G.stage2_params(m), seed=seed,
                tag=f"{tag} {br} S2")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    return np.where(is_f, 0.4 * P["futures"] + 0.6 * P["all"],
                    0.4 * P["regular"] + 0.6 * P["all"])


if __name__ == "__main__":
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    MLPF = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))
    d = F.build(DATA, VS=VS)

    # 지인 전처리 행 순서를 우리 쪽에 맞춘다
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
    Xf, order_f, cols_f = friend_matrix()
    pos = pd.Series(np.arange(len(order_f)), index=order_f)
    Xf = Xf[pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()]
    ci_f = [cols_f.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
    G.log(f"\n  우리 {d['X44'].shape}  지인 {Xf.shape}   "
          f"열 이름 동일: {list(d['F44']) == cols_f}")

    res = {}
    for name, X, ci in (("우리 전처리", d["X44"], d["cat_idx"]),
                        ("지인 전처리", Xf, ci_f)):
        t0 = time.time()
        ps = [run(name, X, ci, sd) for sd in SEEDS]
        solos = [F.best_shift(p[FULL], yv[FULL])[0] for p in ps]
        r = np.mean(ps, 0)
        solo = F.best_shift(r[FULL], yv[FULL])[0]
        bl = F.best_shift((0.10 * CB + 0.30 * MLPF + 0.60 * r)[FULL], yv[FULL])[0]
        np.save(os.path.join(OUT, f"tabmfeat_{'ours' if '우리' in name else 'friend'}.npy"), r)
        res[name] = (solo, bl)
        G.log(f"\n  [{name}]  시드평균 단독 {solo:7.1f}   제출비중 {bl:7.1f}"
              f"   개별 [{', '.join(f'{v:.1f}' for v in solos)}]   {time.time()-t0:.0f}s")

    a, b = res["우리 전처리"], res["지인 전처리"]
    G.log(f"\n  우리 - 지인   단독 {a[0]-b[0]:+.1f}   제출비중 {a[1]-b[1]:+.1f}")
    G.log("  submit_14(1041)는 지인 전처리로 만들었다. 우리 쪽이 확실히 높으면")
    G.log("  그것만으로 재현 격차 6점의 일부가 설명된다.")
