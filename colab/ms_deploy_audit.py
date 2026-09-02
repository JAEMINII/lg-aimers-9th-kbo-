# -*- coding: utf-8 -*-
"""감사판 MultiState(58피처, K32 d512)를 전 구간으로 학습해 배치용으로 내보낸다.

근거 (2026-08-30 구성원 평가, 3원 재조정 혼합 위에 확률혼합)
    VS=2022 +1.3 (w=0.10~0.15)   VS=2023 1군 +2.9~+3.9   VS=2024 +8.2~+11.7
    세 폴드 전부 양수. 2022에서 고른 4원 가중 (cb.35/tab.15/din.35/ms.15) 이
    2024에서 +11.9 로 검증됐다. DIN(두 폴드)보다 근거가 두껍다.

동료의 export_multistate.py 를 감사판 피처(AUDIT_MODE=all)로 확장한 것.
    Xin = [x44(44) | 수치 extras(12) | c4(1) | hand_pair(1)]
    범주 = 기존 9 + c4 + hand_pair = 11
학습은 multistate_softmax.train_model 그대로 (STATE_W .90 / DIRECT_W .35 / BRIER_W .25).
저장은 .pt (preprocessor 메타 + state_dict). npz 변환과 numpy 추론은 로컬에서 한다.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("AUDIT_MODE", "all")
os.environ.setdefault("K", "32")
os.environ.setdefault("DBLOCK", "512")
ROOT = Path("/root")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "aimers"))
DATA = Path(os.environ.get("AIMERS_DATA", "/root/open (1)/data"))
OUT = Path(os.environ.get("MSA_OUT", "/root/msa_export"))
OUT.mkdir(parents=True, exist_ok=True)

import features44 as FF                                         # noqa: E402
import multistate_softmax as M                                  # noqa: E402
import multistate_auditfeat as A                                # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402


def recover_state6(df):
    """4범주의 other-실패를 ball/strike 로 가른다 (2024 단독 +14.8 / 혼합 +6.5).
    0=성공, 1=rev, 2=mid, 3=other-ball, 4=other-strike, -1=분할불가(마스크 제외)."""
    s = M.recover_state(df).copy()
    pid = df.pitcher_id.to_numpy()
    n = df.asof_pitcher_n.fillna(0.).to_numpy(float)
    nxt = (pid[1:] == pid[:-1]) & np.isclose(np.diff(n), 1., atol=1e-8)
    src = np.flatnonzero(nxt) + 1
    dst = src - 1
    cum = df["asof_pitcher_ball_rate"].fillna(0.).to_numpy(float) * n
    inc = cum[src] - cum[dst]
    lab = np.rint(inc)
    good = (np.abs(inc - lab) < .25) & ((lab == 0) | (lab == 1))
    ball = np.full(len(df), np.nan)
    ball[dst[good]] = lab[good]
    out = s.copy()
    m3 = s == 3
    out[m3 & (ball == 1)] = 3
    out[m3 & (ball == 0)] = 4
    out[m3 & ~np.isfinite(ball)] = -1
    return out


def make6(n_num, cards, seed):
    from rtdl_num_embeddings import LinearReLUEmbeddings
    from tabm import TabM
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return TabM.make(n_num_features=n_num,
                     cat_cardinalities=[int(c) for c in cards], d_out=6,
                     num_embeddings=LinearReLUEmbeddings(n_num, d_embedding=16),
                     arch_type="tabm", k=32, n_blocks=3,
                     d_block=M.DBLOCK, dropout=0.1).to(M.DEVICE)



def main():
    raw = PP.sort_by_row_id(pd.read_csv(DATA / "train.csv", encoding="utf-8-sig"))
    y = raw.control_success.to_numpy(np.float32)
    season = raw.season.to_numpy(np.int16)
    isf = raw.game_type.astype(str).to_numpy() == "F"
    old = season <= 2022
    st6 = os.environ.get("MSA_STATE6", "0") == "1"
    aux = recover_state6(raw) if st6 else M.recover_state(raw)
    built = FF.build(str(DATA), VS=2025)
    X44 = built["X44"].astype(np.float32)
    names = list(built["F44"])
    xnum, xnum_names, xcat, xcat_names = A.audit_features(raw, X44, names)
    if os.environ.get("MSA_C4V2", "0") == "1":
        # 1군 경계를 2023|2024 로 (공식 데이터 관측: 1군 성공률 .5031->.4897 단차).
        # 퓨처스 경계는 2022|2023 유지. 값 집합 {0,1,2,3} 불변 -> 추론 호환.
        old_reg = season <= 2023
        c4 = np.where(old & isf, 0.0, np.where(old_reg & ~isf, 1.0,
                      np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
    else:
        c4 = np.where(old & isf, 0.0, np.where(old & ~isf, 1.0,
                      np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
    Xin = np.concatenate([X44, xnum, c4, xcat], axis=1)
    ci = [names.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
    c4_index = X44.shape[1] + xnum.shape[1]
    cat_idx = ci + [c4_index] + list(range(c4_index + 1, Xin.shape[1]))
    if os.environ.get("MSA_IDFREE", "0") in ("1", "pitcher"):
        # cold-행 전용: 투수/타자 임베딩 제거 (id 는 연도 불안정성의 원천)
        if os.environ.get("MSA_IDFREE") == "pitcher":
            dropi = {names.index("pitcher_id")}
        else:
            dropi = {names.index("pitcher_id"), names.index("batter_id")}
        keepj = [j for j in range(Xin.shape[1]) if j not in dropi]
        o2n = {j: k for k, j in enumerate(keepj)}
        Xin = Xin[:, keepj]
        cat_idx = [o2n[j] for j in cat_idx if j not in dropi]
    mask = np.ones(len(y), bool)
    Xn, Xc, cards = G.prep(Xin, mask, cat_idx)
    G.XN, G.XC, G.cards = torch.from_numpy(Xn), torch.from_numpy(Xc), cards
    idx = np.arange(len(y), dtype=np.int64)
    w = np.where(isf & old, 0.1, 1.0).astype(np.float32)
    dec = float(os.environ.get("MSA_DECAY", "1.0"))
    if dec > 1.0:
        # recency 가중 — 두 폴드 검증 (혼합 2024 +2.9 / 2022 +2.1)
        w = (w * dec ** (season.astype(np.float32) - 2019.0)).astype(np.float32)

    all_names = names + xnum_names + ["abs_regime"] + xcat_names
    if os.environ.get("MSA_IDFREE", "0") in ("1", "pitcher"):
        all_names = [all_names[j] for j in keepj]
    ni = [j for j in range(Xin.shape[1]) if j not in set(cat_idx)]
    numcols = [all_names[j] for j in ni]
    catcols = [all_names[j] for j in cat_idx]
    catvals = {c: sorted(set(Xin[mask, j][np.isfinite(Xin[mask, j])].tolist()))
               for c, j in zip(catcols, cat_idx)}
    med = np.nanmedian(Xin[:, ni], 0)
    med = np.where(np.isfinite(med), med, 0.0)
    filled = np.where(np.isnan(Xin[:, ni]), med, Xin[:, ni])
    mu, sd = filled.mean(0), filled.std(0) + 1e-6
    missing = [numcols[k] for k, j in enumerate(ni) if np.isnan(Xin[:, j]).any()]
    prep = {"feature_names": all_names, "cat_cols": catcols, "num_cols": numcols,
            "cat_values": catvals,
            "medians": dict(zip(numcols, med.tolist())),
            "means": dict(zip(numcols, mu.tolist())),
            "stds": dict(zip(numcols, sd.tolist())),
            "missing_cols": missing,
            "audit_mode": os.environ["AUDIT_MODE"],
            "extras_num": xnum_names, "extras_cat": xcat_names}
    M.log(f"MSA export rows={len(idx):,} X={Xin.shape} n_num={Xn.shape[1]} "
          f"cards={list(cards)} K={M.K} d={M.DBLOCK} ep={M.EPOCHS}")
    s2 = os.environ.get("MSA_S2", "0") == "1"
    for seed in M.SEEDS:
        t0 = time.time()
        m0 = (make6(Xn.shape[1], cards, seed) if st6
              else M.make_model(Xn.shape[1], cards, seed))
        m = M.train_model(m0, idx, y, aux, w,
                          seed, f"MSA-EXPORT s{seed}")
        if s2:
            # 마지막 시즌(2024) 1에폭 lr 2e-4 — 관문 검증: 혼합 2024 +3.1 / 2022 +2.3
            s2i = idx[season[idx] == 2024]
            lr0, ep0 = M.LR, M.EPOCHS
            M.LR, M.EPOCHS = 2e-4, 1
            m = M.train_model(m, s2i, y, aux, w, seed + 1,
                              f"MSA-EXPORT s{seed} S2")
            M.LR, M.EPOCHS = lr0, ep0
        payload = {"config": {"k": M.K, "arch_type": "tabm", "n_blocks": 3,
                              "d_block": M.DBLOCK, "num_embeddings": "linear_relu",
                              "loss": "multistate_direct_aux_auditfeat"},
                   "preprocessor": prep,
                   "model_state": {k: v.detach().cpu()
                                   for k, v in m.state_dict().items()}}
        torch.save(payload, OUT / f"msa_seed{seed}.pt")
        M.log(f"saved msa_seed{seed}.pt ({time.time()-t0:.0f}s)")
        del m
        torch.cuda.empty_cache()
    print(json.dumps({"out": str(OUT), "n_num": int(Xn.shape[1]),
                      "cards": [int(c) for c in cards]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
