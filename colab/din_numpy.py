# -*- coding: utf-8 -*-
"""DIN-lite 의 numpy 전용 추론. 제출 패키지는 numpy/pandas 만 쓴다.

torch 쪽 정의 (ctr_zoo.DINBST, mode="din") 를 그대로 옮긴 것이다.
    e      = concat_i emb_i(xc[:, i])                       (B, 10*16)
    s      = concat(se_state(sq_state), se_cell(sq_cell))   (B, K, 32)
    q      = q_cell(cur)                                    (B, 16)
    a      = att(concat(s, q_expand, s * tile(q,2)))        (B, K, 1)
    a      = softmax(mask_fill(a, -1e9), axis=1)
    pooled = sum_k a * s                                    (B, 32)
    logit  = mlp(concat(xn, e, pooled))                     (B,)

전처리도 학습 때(ctr_zoo.build_inputs) 만든 표를 그대로 쓴다. 표는 din_meta.npz 에
들어 있다 — 학습과 추론이 같은 표를 보게 하려는 것이다.
"""
import numpy as np


def din_prep(X45, M):
    """X45: (n, 45) float64 — PP.transform_features 44열 + abs_regime."""
    ci, ni = M["ci"].astype(np.int64), M["ni"].astype(np.int64)
    Xc = np.zeros((len(X45), len(ci)), np.int64)
    for a, j in enumerate(ci):
        u = np.asarray(M[f"uval_{a}"], np.float64)
        col = X45[:, j]
        p = np.clip(np.searchsorted(u, col), 0, max(len(u) - 1, 0))
        Xc[:, a] = np.where((p < len(u)) & (u[p] == col), p + 1, 0)
    Xn = X45[:, ni].astype(np.float64)
    miss = np.isnan(Xn)
    Xn = np.where(miss, M["med"], Xn)
    Xn = ((Xn - M["mu"]) / M["sd"]).astype(np.float32)
    hn = M["hn"].astype(bool)
    if hn.any():
        Xn = np.concatenate([Xn, miss[:, hn].astype(np.float32)], 1)
    return Xn, Xc


def din_seq(pitcher_id, balls, strikes, batter_hand, M):
    """(투수 -> 2025 슬라이스) 조회 + 현재 행의 상황 질의."""
    pids = M["pids"].astype(np.int64)
    q = np.asarray(pitcher_id, np.int64)
    p = np.clip(np.searchsorted(pids, q), 0, max(len(pids) - 1, 0))
    hit = (p < len(pids)) & (pids[p] == q)
    idx = np.where(hit, p, 0)
    S = M["dep_state"][idx].astype(np.int64)
    C = M["dep_cell"][idx].astype(np.int64)
    Mk = M["dep_mask"][idx].astype(bool)
    # 처음 보는 투수는 이력 없음으로 둔다 (패딩 한 칸만 열어 둔다)
    S[~hit] = 0; C[~hit] = 0
    Mk[~hit] = False; Mk[~hit, 0] = True
    cell = np.clip(np.asarray(balls, np.int64) * 3
                   + np.asarray(strikes, np.int64), 0, 11) * 2 \
        + np.asarray(batter_hand, np.int64)
    cur = np.clip(cell, 0, 24) + 1
    return S, C, Mk, cur


def _relu(x):
    return np.maximum(x, 0.0)


def din_forward(Xn, Xc, S, C, Mk, cur, W, chunk=8192):
    out = np.empty(len(Xn), np.float64)
    ne = sum(1 for k in W.files if k.startswith("emb__"))
    for a in range(0, len(Xn), chunk):
        b = min(a + chunk, len(Xn))
        e = np.concatenate([W[f"emb__{i}__weight"][Xc[a:b, i]]
                            for i in range(ne)], 1)                  # (B,160)
        s = np.concatenate([W["se_state__weight"][S[a:b]],
                            W["se_cell__weight"][C[a:b]]], 2)        # (B,K,32)
        q = W["q_cell__weight"][cur[a:b]]                            # (B,16)
        K = s.shape[1]
        qe = np.repeat(q[:, None, :], K, 1)                          # (B,K,16)
        q2 = np.concatenate([qe, qe], 2)[:, :, :s.shape[2]]          # (B,K,32)
        h = np.concatenate([s, qe, s * q2], 2)                       # (B,K,80)
        h = _relu(h @ W["att__0__weight"].T + W["att__0__bias"])
        sc = h @ W["att__2__weight"].T + W["att__2__bias"]           # (B,K,1)
        sc = np.where(Mk[a:b][:, :, None], sc, -1e9)
        sc = sc - sc.max(1, keepdims=True)
        ex = np.exp(sc)
        att = ex / ex.sum(1, keepdims=True)
        pooled = (att * s).sum(1)                                    # (B,32)
        z = np.concatenate([Xn[a:b], e, pooled], 1)
        z = _relu(z @ W["mlp__0__weight"].T + W["mlp__0__bias"])
        z = _relu(z @ W["mlp__3__weight"].T + W["mlp__3__bias"])
        z = (z @ W["mlp__6__weight"].T + W["mlp__6__bias"]).ravel()
        out[a:b] = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    return out
