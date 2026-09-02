# -*- coding: utf-8 -*-
"""flat MLP 순수 numpy 추론.

torch 없이 돌아간다. 제출 script.py 에 인라인할 수 있게 이 파일만으로 완결된다.

구조 (mlp_gpu.MLP_PLR 과 동일)
    PLR 수치임베딩   2*pi*w*x -> [cos, sin] -> 피처별 Linear(96->24) + ReLU
    범주임베딩       Embedding 9개 (각 8차원)
    본체            concat(51*24 + 9*8 = 1296) -> 128 -> 256 -> 128 -> 1 -> sigmoid
    최종            시드 3개 확률 평균

rtdl_num_embeddings.PeriodicEmbeddings 의 forward 순서
    x = 2*pi * weight * x[..., None]
    x = cat([cos(x), sin(x)], -1)          <- cos 가 먼저
    x = linear(x)                          <- 피처별 (2k -> d)
    x = relu(x)
"""
import numpy as np


def _plr(xn, per_w, lin_w, lin_b):
    """(n, F) -> (n, F*d).  피처마다 독립적인 주기임베딩 + 선형 + ReLU."""
    n, nf = xn.shape
    d = lin_w.shape[-1] if lin_w.ndim == 3 else lin_w.shape[0]
    out = np.empty((n, nf, d), np.float32)
    for f in range(nf):
        z = (2.0 * np.pi * per_w[f])[None, :] * xn[:, f:f + 1]
        p = np.concatenate([np.cos(z), np.sin(z)], 1).astype(np.float32)
        out[:, f, :] = p @ lin_w[f] + lin_b[f]
    np.maximum(out, 0, out=out)   # 위치인수 3개는 numpy 2.4 에서 경고
    return out.reshape(n, nf * d)


def predict_one(Xn, Xc, z, chunk=16384):
    """z: np.load 한 시드 하나의 npz.  반환 (n,) 확률."""
    per_w = z["num_emb__periodic__weight"]
    lin_w = z["num_emb__linear__weight"]
    lin_b = z["num_emb__linear__bias"]
    cat_w = [z[f"cat_embs__{j}__weight"] for j in range(9)]
    W = [(z["body__0__weight"].T, z["body__0__bias"]),
         (z["body__2__weight"].T, z["body__2__bias"]),
         (z["body__4__weight"].T, z["body__4__bias"])]
    hw, hb = z["head__weight"].T, z["head__bias"]
    out = np.empty(len(Xn), np.float64)
    for a in range(0, len(Xn), chunk):
        b = min(a + chunk, len(Xn))
        h = _plr(Xn[a:b], per_w, lin_w, lin_b)
        parts = [h] + [cat_w[j][Xc[a:b, j]] for j in range(9)]
        x = np.concatenate(parts, 1)
        for w, bb in W:
            x = np.maximum(x @ w + bb, 0)
        lg = (x @ hw + hb).ravel()
        out[a:b] = 1.0 / (1.0 + np.exp(-lg.astype(np.float64)))
    return out


def predict(Xn, Xc, archives, chunk=16384):
    """시드별 확률의 평균. 개별 시드를 고르면 운을 고르는 것이라 평균이 맞다."""
    return np.mean([predict_one(Xn, Xc, z, chunk) for z in archives], 0)


def prep(X, stats):
    """학습 때 저장한 통계로 전처리. 통계는 전부 학습 구간에서 뽑은 것이다."""
    ci = np.asarray(stats["cat_idx"])
    ni = np.asarray([j for j in range(X.shape[1]) if j not in set(ci.tolist())])
    Xc = np.zeros((len(X), len(ci)), np.int64)
    for a in range(len(ci)):
        keys = stats[f"catkey_{a}"]
        vals = stats[f"catval_{a}"]
        q = X[:, ci[a]].astype(np.float64)
        pos = np.clip(np.searchsorted(keys, q), 0, max(len(keys) - 1, 0))
        hit = keys[pos] == q if len(keys) else np.zeros(len(X), bool)
        Xc[:, a] = np.where(hit, vals[pos], 0)
    Xn = X[:, ni].astype(np.float64)
    miss = np.isnan(Xn)
    Xn = np.where(miss, stats["med"], Xn)
    Xn = ((Xn - stats["mu"]) / stats["sd"]).astype(np.float32)
    hn = stats["has_nan"]
    if hn.any():
        Xn = np.concatenate([Xn, miss[:, hn].astype(np.float32)], 1)
    return Xn, Xc
