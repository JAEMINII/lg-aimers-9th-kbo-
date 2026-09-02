# -*- coding: utf-8 -*-
"""flat MLP 를 '지인 preprocess' 피처로 학습한다.

왜 바꾸나
    우리 features44 와 지인 preprocess 는 이름·순서가 같지만 값이 조금 다르다.
        plat_dev   최대 2.5e-02 (평균 2.1e-3)   <- 플래툰 수축 계산이 다름
        cm_rel     최대 6.4e-04
        나머지 40개는 float32 오차 수준
    TabM 은 지인 피처로 학습됐다. MLP 도 같은 피처로 맞추면 제출 script.py 에서
    피처 파이프라인을 하나만 돌리면 되고, 학습/추론 불일치 위험도 사라진다.

구성은 관문에서 확정된 것 그대로
    전체 시즌 균등 학습 (시즌가중 없음), 6 epoch, 시드 3개 평균
    loss = 0.5 BCE + 0.5 Brier

사용법
    python mlp_friendfeat.py 2024     관문 검증 (2019~2023 학습 -> 2024 채점)
    python mlp_friendfeat.py 2025     제출용 (2019~2024 전체 학습)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, "/workspace/aimers")
OUT = "/workspace/aimers/out"
EX = os.path.join(OUT, "mlp_ff")
os.makedirs(EX, exist_ok=True)
D = "/workspace/aimers/data"
SEEDS = (42, 1, 777)
EPOCHS = 6

import mlp_gpu as M                                          # noqa: E402
from train_chan_3 import preprocess as PP                    # noqa: E402


def build(VS):
    """지인 경로로 44피처를 만든다. 룩업은 학습 구간에서만."""
    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(D, "train.csv"),
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr[tr.season < VS])
    X = PP.transform_features(tr, hist, train_mode=True)
    y = tr[PP.TARGET].to_numpy(dtype=np.float32)
    season = tr["season"].to_numpy()
    is_f = tr["game_type"].astype(str).to_numpy() == "F"
    return X, y, season, is_f, hist


def prep_stats(X, m_tr, cat_names):
    """TabM 과 같은 규약. 통계는 학습 구간에서만 뽑는다."""
    cols = list(X.columns)
    ci = [cols.index(c) for c in cat_names]
    A = X.to_numpy(dtype=np.float64)
    ni = [j for j in range(len(cols)) if j not in set(ci)]
    st = {"cat_idx": np.asarray(ci), "num_idx": np.asarray(ni),
          "columns": np.array(cols, dtype="U40")}
    Xc = np.zeros((len(A), len(ci)), np.int64)
    for a, j in enumerate(ci):
        vals = np.unique(A[m_tr, j])
        vals = vals[~np.isnan(vals)]
        st[f"catkey_{a}"] = vals.astype(np.float64)
        st[f"catval_{a}"] = np.arange(1, len(vals) + 1, dtype=np.int64)
        pos = np.clip(np.searchsorted(vals, A[:, j]), 0, max(len(vals) - 1, 0))
        hit = (vals[pos] == A[:, j]) if len(vals) else np.zeros(len(A), bool)
        Xc[:, a] = np.where(hit, pos + 1, 0)
    Xn = A[:, ni]
    miss = np.isnan(Xn)
    med = np.nanmedian(Xn[m_tr], 0)
    Xn = np.where(miss, med, Xn)
    mu, sd = Xn[m_tr].mean(0), Xn[m_tr].std(0) + 1e-6
    Xn = ((Xn - mu) / sd).astype(np.float32)
    hn = miss[m_tr].any(0)
    st.update(med=med, mu=mu, sd=sd, has_nan=hn)
    if hn.any():
        Xn = np.concatenate([Xn, miss[:, hn].astype(np.float32)], 1)
    cards = np.array([int(st[f"catval_{a}"].max()) + 1 for a in range(len(ci))])
    return Xn, Xc, cards, st


if __name__ == "__main__":
    VS = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    X, y, season, is_f, hist = build(VS)
    m_tr = season < VS
    Xn, Xc, cards, st = prep_stats(X, m_tr, PP.TABM_CATEGORICAL_FEATURES)
    np.savez_compressed(os.path.join(EX, f"prep_vs{VS}.npz"), **st)
    M.XN = torch.from_numpy(Xn)
    M.XC = torch.from_numpy(Xc)
    M.YY = torch.from_numpy(y)
    tr_idx = np.where(m_tr)[0]
    print(f"VS={VS}  학습 {len(tr_idx):,}  수치 {Xn.shape[1]}열  범주 {Xc.shape[1]}열  "
          f"카디널리티 {cards.tolist()}", flush=True)

    preds = []
    for sd_ in SEEDS:
        torch.manual_seed(sd_)
        m = M.MLP_PLR(Xn.shape[1], cards).to(M.DEV)
        M.train(m, tr_idx, EPOCHS, 1e-3, tag=f"vs{VS} s{sd_}")
        sdict = {k: v.detach().cpu().numpy().astype(np.float32)
                 for k, v in m.state_dict().items()}
        np.savez_compressed(os.path.join(EX, f"vs{VS}_seed{sd_}.npz"),
                            **{k.replace(".", "__"): v for k, v in sdict.items()})
        if VS == 2024:
            preds.append(M.predict(m, np.where(~m_tr)[0]))
        del m
        torch.cuda.empty_cache()
    json.dump(dict(n_num=int(Xn.shape[1]), cards=[int(c) for c in cards],
                   seeds=list(SEEDS), epochs=EPOCHS, vs=VS,
                   cat_names=list(PP.TABM_CATEGORICAL_FEATURES),
                   columns=list(X.columns)),
              open(os.path.join(EX, f"vs{VS}_meta.json"), "w"),
              ensure_ascii=False, indent=1)

    if VS == 2024:
        p = np.mean(preds, 0)
        np.save(os.path.join(OUT, "mlpff_gate.npy"), p)
        from scipy.optimize import minimize_scalar
        gate = np.where(~m_tr)[0]
        fv, yv = is_f[gate], y[gate].astype(np.float64)

        def sc(q, msk):
            t = yv[msk]
            r = t.mean()
            def b(c):
                v = np.clip(q[msk], 1e-6, 1 - 1e-6)
                v = 1 / (1 + np.exp(-(np.log(v / (1 - v)) + c)))
                return 100000 * (1 - ((v - t) ** 2).mean() / (r * (1 - r)))
            rr = minimize_scalar(lambda c: -b(c), bounds=(-.3, .3), method="bounded")
            return -rr.fun
        CB = np.load(os.path.join(SC, "cb_gate.npy"))
        mA = np.ones(len(gate), bool)
        print(f"\n  지인피처 flat MLP 시드3   단독 1군 {sc(p, ~fv):.1f} / "
              f"전체 {sc(p, mA):.1f}   CB상관 {np.corrcoef(p[~fv], CB[~fv])[0,1]:.4f}")
        for w in (0.20, 0.25, 0.30, 0.35, 0.40):
            q = (1 - w) * CB + w * p
            print(f"    비중 {w:.2f}   1군 {sc(q, ~fv):8.1f}   전체 {sc(q, mA):8.1f}")
        print(f"  (우리피처 판본은 전체 937.2 였다. 기존 MLP 는 911.2)")
