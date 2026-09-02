# -*- coding: utf-8 -*-
"""HistGB 트리(npz)를 numpy 로 순회한다. 제출본 script.py 에 그대로 들어간다.

sklearn 의 HistGradientBoostingClassifier 예측을 재현한다.
  분기    값 <= num_threshold 면 왼쪽
  결측    missing_go_to_left 플래그
  범주형  없음 (내보낼 때 검증)
  raw     baseline + 잎값 합,  proba = sigmoid(raw)

submit_jaemin_2 의 predict_numpy 와 순회 방식은 같되, 모델 간 평균을 내지 않고
브랜치별 확률을 (n, n_models) 로 돌려준다. 라우팅을 밖에서 하기 때문이다.
"""
import numpy as np


def predict_branches(X, z, chunk=4096):
    """X: (n, F) float64.  반환: (n, n_models) 브랜치별 확률."""
    T, M = z["feat"].shape
    fl = z["feat"].ravel().astype(np.int64)
    tl = z["thr"].ravel()
    ll = z["left"].ravel().astype(np.int64)
    rl = z["right"].ravel().astype(np.int64)
    lfl = z["leaf"].ravel().astype(bool)
    mgl_ = z["mgl"].ravel().astype(bool)
    vl = z["val"].ravel()
    base = z["baselines"]
    depth = int(z["max_depth"])
    n_models = int(z["n_models"])
    per = T // n_models
    root = np.arange(T, dtype=np.int64) * M

    n, Fw = X.shape
    Xf = np.ascontiguousarray(X, dtype=np.float64).ravel()
    out = np.zeros((n, n_models), np.float64)
    off0 = np.repeat(np.arange(chunk, dtype=np.int64) * Fw, T)

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        m = e - s
        idx = np.tile(root, m)
        off = off0[:m * T] + s * Fw
        act = np.arange(m * T, dtype=np.int64)
        for _ in range(depth + 1):
            nd = idx[act]
            keep = ~lfl[nd]
            if not keep.any():
                break
            act = act[keep]
            nd = nd[keep]
            v = Xf[off[act] + fl[nd]]
            go_left = np.where(np.isnan(v), mgl_[nd], v <= tl[nd])
            idx[act] = np.where(go_left, ll[nd], rl[nd])
        raw = vl[idx].reshape(m, n_models, per).sum(axis=2)
        out[s:e] = 1.0 / (1.0 + np.exp(-(raw + base[None, :])))
    return out


def route(pb, is_f, w_branch=0.4):
    """0.6 x all + 0.4 x (경기유형별). 989 를 만든 라우팅이다.

    pb 의 열 순서는 npz 의 branches 와 같다: all / futures / regular.
    """
    pa, pf, pr = pb[:, 0], pb[:, 1], pb[:, 2]
    sp = np.where(is_f, pf, pr)
    return (1.0 - w_branch) * pa + w_branch * sp
