# -*- coding: utf-8 -*-
"""학습한 PyTorch 모델과 배치 순전파가 같은 값을 내는가.

안 본 자리
    학습     OfficialTabMEstimator (PyTorch, tabm 라이브러리)
    내보내기 export_one 이 체크포인트 -> npz 로 가중치를 뽑음
    추론     script.py 의 _tabm_forward 가 그 npz 로 순전파를 **다시 구현**

    GPU 포팅 때 numpy <-> torch 를 맞췄지만(최대차 1.19e-07) 그건 **둘 다
    script.py 의 재구현**이었다. 원본 모델과는 한 번도 안 맞춰봤다.

    재구현이 어긋나면 조용히 틀린 예측을 낸다. plat_dev 와 같은 부류다 —
    학습한 것과 배치되는 것이 다른 종류의 결함.

특히 의심스러운 대목
    _tabm_forward 의 block 0 은 원핫을 만들지 않고 범주 열만 골라 더한다.
        h = (nr * r) @ W[:, :nd].T
        for j: h += r[:, off:off+card][:, idx].T[:, :, None] * W[...].T[:, None, :]
    '2,489차원 원핫을 만들지 않는 정확한 등가' 라고 주석에 적혀 있는데
    그 등가성을 확인한 기록이 없다.

    그리고 num_embeddings 가 linear_relu 일 때만 처리한다. 다른 값이면 에러를
    내니 조용히 틀릴 일은 없다.

무엇을 하나
    체크포인트를 PyTorch 로 불러 그대로 추론하고, 같은 행을 npz 경로로도
    추론해서 최대차를 본다. 1e-5 를 넘으면 결함이다.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/workspace/aimers")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "colab"))

from train_chan_3 import preprocess as PP                       # noqa: E402
from train_chan_3.official_tabm import OfficialTabMEstimator    # noqa: E402

DATA = ROOT / "data"
CKPT = ROOT / "art_platfix" / "all_s42.pt"
NPZ = ROOT / "npz_platfix" / "all_s42.npz"
N = 20000


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def npz_forward(nums, cats, meta, ar, chunk=4096):
    """submit_30 의 _tabm_forward 를 그대로 옮긴 것."""
    w = ar["num_embedding_weight"]
    b = ar["num_embedding_bias"]
    nr = np.maximum(nums[:, :, None] * w[None] + b[None], 0.0)
    nr = nr.reshape(len(nums), -1)
    k = int(meta["k"])
    cards = list(meta["cat_cardinalities"])
    nd = nr.shape[1]
    offs = np.cumsum([0] + cards[:-1]) + nd
    out = []
    for s in range(0, len(nr), chunk):
        cn, cc = nr[s:s + chunk], cats[s:s + chunk]
        h = None
        for blk in range(int(meta["n_blocks"])):
            W = ar[f"block_{blk}_weight"]
            r = ar[f"block_{blk}_r"]
            sc_ = ar[f"block_{blk}_s"]
            bb = ar[f"block_{blk}_bias"]
            if blk == 0:
                h = np.matmul(cn[:, None, :] * r[None, :, :nd], W[:, :nd].T)
                for j, c in enumerate(cards):
                    o = int(offs[j])
                    idx = cc[:, j]
                    sr = r[:, o:o + c][:, idx].T
                    sw = W[:, o:o + c][:, idx].T
                    h += sr[:, :, None] * sw[:, None, :]
            else:
                h = np.matmul(h * r[None], W.T)
            h = np.maximum(h * sc_[None] + bb[None], 0.0)
        lg = np.einsum("bki,kio->bko", h, ar["output_weight"]) \
            + ar["output_bias"][None]
        out.append(sig(lg[:, :, 0]).mean(axis=1))
    return np.concatenate(out)


if __name__ == "__main__":
    tr = PP.sort_by_row_id(pd.read_csv(DATA / "train.csv",
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr)
    X = PP.transform_features(tr, hist, train_mode=True)
    # plat_dev 를 as-of 로 (submit_30 학습과 같게)
    import features44 as F
    d = F.build(str(DATA), VS=2025, return_frame=True)
    j = list(d["F44"]).index("plat_dev")
    s = pd.Series(d["X44"][:, j].astype(np.float64),
                  index=d["frame"]["row_id"].to_numpy())
    X["plat_dev"] = s.reindex(tr["row_id"].to_numpy()).to_numpy(np.float64)

    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(X), N, replace=False))
    Xs = X.iloc[sel].reset_index(drop=True)

    # ---- 경로 A: 학습한 PyTorch 모델 그대로
    model = OfficialTabMEstimator.load(CKPT, device="cuda")
    pa = np.asarray(model.predict_proba(Xs), dtype=np.float64)
    if pa.ndim == 2:
        pa = pa[:, 1]
    print(f"  PyTorch 경로  {len(pa):,}행  평균 {pa.mean():.6f}")

    # ---- 경로 B: 내보낸 npz + script.py 의 재구현
    z = np.load(NPZ, allow_pickle=False)
    meta = json.loads(str(z["metadata"].item()))
    meta.setdefault("cat_cardinalities", meta.get("cards"))
    exp = list(meta["feature_names"])
    Xf = Xs[exp]
    cat_cols = list(meta["cat_cols"])
    num_cols = list(meta["num_cols"])
    cats = np.zeros((len(Xf), len(cat_cols)), np.int64)
    for jj, col in enumerate(cat_cols):
        mp = {v: i + 1 for i, v in enumerate(meta["cat_values"][col])}
        cats[:, jj] = Xf[col].map(mp).fillna(0).astype("int64").to_numpy()
    numeric = Xf[num_cols].apply(pd.to_numeric, errors="coerce")
    parts = []
    for col in num_cols:
        v = numeric[col].astype("float64").fillna(float(meta["medians"][col]))
        parts.append(((v - float(meta["means"][col]))
                      / float(meta["stds"][col])).to_numpy("float32"))
    for col in meta["missing_cols"]:
        parts.append(numeric[col].isna().to_numpy("float32"))
    nums = np.column_stack(parts).astype("float32")
    pb = npz_forward(nums, cats, meta, z).astype(np.float64)
    print(f"  npz 경로      {len(pb):,}행  평균 {pb.mean():.6f}")

    dd = np.abs(pa - pb)
    print(f"\n  최대차 {dd.max():.3e}   평균차 {dd.mean():.3e}")
    print(f"  1e-5 초과 {int((dd > 1e-5).sum()):,}개 / {len(dd):,}")
    print(f"  상관 {np.corrcoef(pa, pb)[0,1]:.10f}")
    if dd.max() < 1e-5:
        print("\n  일치한다. 내보내기와 재구현은 깨끗하다.")
    else:
        print("\n  **어긋난다.** 학습한 모델과 배치되는 순전파가 다르다 —")
        print("  plat_dev 와 같은 종류의 결함이고, 크기는 이 차이만큼이다.")
