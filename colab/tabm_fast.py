# -*- coding: utf-8 -*-
"""TabM numpy 순전파를 빠르게 다시 쓰고, 원본과 값이 같은지 대조한다.

왜
    추론 예산 10분 중 순전파가 94%다. 그 제약 때문에 F2B_SEEDS=(42,) 로 묶여 있고
    시드 평균도 CatBoost 비중도 못 늘린다. 그런데 실측 처리량이 11.6 GFLOPS 다 —
    8코어 float32 면 50~100 이 나와야 한다. 모델을 안 바꾸고 되찾을 수 있는 예산이다.

원본이 흘리는 곳
    block 0    h = matmul(nums[:,None,:] * r[None,:,:D], W[:,:D].T)
               (B, k, D) 짜리 임시 배열을 만든다. B=4096, k=32, D=576 이면
               7,550만 원소다. 곱셈 자체보다 이 쓰기/읽기가 비싸다.
    block 1,2  h * r[None,:,:] 이 또 (B, k, 256) 을 만든다.

고치는 법 — 수학은 그대로다
    (x * r_k) @ W.T  =  x @ (r_k[:,None] * W.T)     r 은 입력축 elementwise 라
    그래서 RW[n, k, o] = r[k,n] * W[o,n] 을 **모델당 한 번** 미리 만들고
    (n, k*o) 로 펴서 한 방의 2D GEMM 으로 끝낸다. 임시 배열이 사라진다.
    block 1,2 도 같은 방식으로 r 을 가중치에 접어 넣는다.

    범주형 항은 원본과 동일하게 유지한다 (one-hot 을 안 펴는 것도 그대로).
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd


def make_fast(metadata, archive):
    """모델당 한 번 호출. r 을 가중치에 접어 넣은 상수들을 돌려준다."""
    k = int(metadata["k"])
    db = int(metadata["d_block"])
    nb = int(metadata["n_blocks"])
    W0 = archive["block_0_weight"]                       # (db, D_total)
    r0 = archive["block_0_r"]                            # (k, D_total)
    cards = list(metadata["cat_cardinalities"])
    pack = {"k": k, "d_block": db, "n_blocks": nb, "cards": cards}
    # block 0 수치부: RW[n, k, o] = r0[k, n] * W0[o, n]
    return pack, W0, r0


def tabm_fast(nums, cats, metadata, archive, chunk_size=8192, cache={}):
    if len(nums) == 0:
        return np.empty(0, np.float32)
    key = id(archive)
    if key not in cache:
        k = int(metadata["k"]); db = int(metadata["d_block"])
        nb = int(metadata["n_blocks"])
        cards = list(metadata["cat_cardinalities"])
        # 수치 임베딩
        ew = archive["num_embedding_weight"].astype(np.float32)
        eb = archive["num_embedding_bias"].astype(np.float32)
        D = ew.shape[0] * ew.shape[1]
        W0 = archive["block_0_weight"].astype(np.float32)
        r0 = archive["block_0_r"].astype(np.float32)
        # RW: (D, k*db)  <- r 을 접어 넣는다
        RW = (r0[:, None, :D] * W0[None, :, :D]).transpose(2, 0, 1).reshape(D, k * db)
        RW = np.ascontiguousarray(RW)
        offs = np.cumsum([0] + cards[:-1]) + D
        # 범주형: (카드 합, k, db) 를 미리 곱해 둔다
        catW = []
        for j, c in enumerate(cards):
            o = int(offs[j])
            rr = r0[:, o:o + c]                          # (k, c)
            ww = W0[:, o:o + c]                          # (db, c)
            catW.append(np.ascontiguousarray(
                (rr[:, None, :] * ww[None, :, :]).transpose(2, 0, 1)))   # (c, k, db)
        later = []
        for b in range(1, nb):
            Wb = archive[f"block_{b}_weight"].astype(np.float32)
            rb = archive[f"block_{b}_r"].astype(np.float32)
            later.append(np.ascontiguousarray(
                rb[:, :, None] * Wb.T[None, :, :]))      # (k, db, db)
        cache[key] = dict(
            ew=ew, eb=eb, D=D, RW=RW, catW=catW,
            sb=[archive[f"block_{b}_s"].astype(np.float32) for b in range(nb)],
            bb=[archive[f"block_{b}_bias"].astype(np.float32) for b in range(nb)],
            later=later, ow=archive["output_weight"].astype(np.float32),
            ob=archive["output_bias"].astype(np.float32), k=k, db=db, nb=nb)
    C = cache[key]
    k, db, nb, D = C["k"], C["db"], C["nb"], C["D"]
    out = np.empty(len(nums), np.float32)
    for s in range(0, len(nums), chunk_size):
        e = min(s + chunk_size, len(nums))
        x = nums[s:e]
        B = e - s
        emb = np.maximum(x[:, :, None] * C["ew"][None] + C["eb"][None], 0.0)
        emb = emb.reshape(B, D)
        h = (emb @ C["RW"]).reshape(B, k, db)            # 한 방의 GEMM
        for j, cw in enumerate(C["catW"]):
            h += cw[cats[s:e, j]]                        # (B, k, db) 게더
        h = np.maximum(h * C["sb"][0][None] + C["bb"][0][None], 0.0)
        for b in range(1, nb):
            h = np.matmul(h.transpose(1, 0, 2), C["later"][b - 1]).transpose(1, 0, 2)
            h = np.maximum(h * C["sb"][b][None] + C["bb"][b][None], 0.0)
        lg = np.einsum("bki,kio->bko", h, C["ow"]) + C["ob"][None]
        z = lg[:, :, 0]
        out[s:e] = (1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))).mean(1)
    return out


if __name__ == "__main__":
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA = os.path.join(ROOT, "open (1)", "data")
    PKG = os.path.join(ROOT, "submit_41")
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    cols = list(pd.read_csv(os.path.join(DATA, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns)
    tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
    pool = tr[tr.season == 2024]
    rng = np.random.default_rng(0)
    sub = pool.iloc[np.sort(rng.choice(len(pool), N, replace=False))][cols] \
              .reset_index(drop=True)
    cwd = os.getcwd(); os.chdir(PKG); sys.path.insert(0, PKG)
    spec = importlib.util.spec_from_file_location("p41", "script.py")
    S = importlib.util.module_from_spec(spec); S.__dict__["__name__"] = "p41"
    spec.loader.exec_module(S)
    import preprocess as PPF
    with open("model/history.json", encoding="utf-8") as f:
        hp = PPF.deserialize_history(json.load(f))
    ordered = PPF.sort_by_row_id(sub)
    Xf = PPF.build_inference_features(ordered, hp)
    isf = ordered["game_type"].astype(str).to_numpy() == "F"
    Xv = np.c_[Xf.to_numpy(dtype=np.float64), np.where(isf, 2.0, 3.0)]
    print(f"  {N:,}행   수치 {Xf.shape[1]}열\n")
    R = 245789 / N
    tot = {"orig": 0.0, "fast": 0.0}
    for br, rows in (("all", np.arange(len(Xf))),
                     ("regular", np.flatnonzero(~isf)),
                     ("futures", np.flatnonzero(isf))):
        z = np.load(os.path.join(PKG, f"model/f2b_{br}_s42.npz"), allow_pickle=False)
        mt = json.loads(str(z["meta"].item())); mt["cat_cardinalities"] = mt["cards"]
        Tn, Tc = S._fm_prep(Xv[rows], z)
        t0 = time.time(); a = S._tabm_forward(Tn, Tc, mt, z, 4096); t1 = time.time()
        b = tabm_fast(Tn, Tc, mt, z)                       # 캐시 채우기
        t2 = time.time(); b = tabm_fast(Tn, Tc, mt, z); t3 = time.time()
        tot["orig"] += t1 - t0; tot["fast"] += t3 - t2
        print(f"  {br:8s} {len(rows):>7,}행   원본 {t1-t0:6.2f}초   "
              f"최적 {t3-t2:6.2f}초  ({(t1-t0)/max(t3-t2,1e-9):4.1f}배)   "
              f"최대차 {np.abs(a-b).max():.3e}")
    print(f"\n  합계   원본 {tot['orig']:6.2f}초 -> 245,789행 {tot['orig']*R/60:5.2f}분")
    print(f"         최적 {tot['fast']:6.2f}초 -> 245,789행 {tot['fast']*R/60:5.2f}분")
    print(f"         되찾은 시간 {(tot['orig']-tot['fast'])*R/60:5.2f}분")
    sys.path.pop(0); os.chdir(cwd)
