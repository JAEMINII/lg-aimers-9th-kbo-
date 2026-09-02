# -*- coding: utf-8 -*-
"""동치 검증용 기준 덤프 — 서버 torch 모델의 직접 예측과 그 입력 행렬 슬라이스.

학습과 같은 코드로 Xin(58열)을 만들고, 30행마다 하나(~49k행)에 대해
torch 예측(direct)과 Xin 슬라이스를 저장한다. 로컬 numpy 순전파가 같은
Xin 에서 같은 값을 내면 변환+순전파가 맞다.
"""
import os
import sys
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
OUT = Path("/root/msa_export")

import features44 as FF                                         # noqa: E402
import multistate_softmax as M                                  # noqa: E402
import multistate_auditfeat as A                                # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

raw = PP.sort_by_row_id(pd.read_csv(DATA / "train.csv", encoding="utf-8-sig"))
y = raw.control_success.to_numpy(np.float32)
season = raw.season.to_numpy(np.int16)
isf = raw.game_type.astype(str).to_numpy() == "F"
old = season <= 2022
built = FF.build(str(DATA), VS=2025)
X44 = built["X44"].astype(np.float32)
names = list(built["F44"])
xnum, xnum_names, xcat, xcat_names = A.audit_features(raw, X44, names)
c4 = np.where(old & isf, 0.0, np.where(old & ~isf, 1.0,
              np.where(isf, 2.0, 3.0))).astype(np.float32)[:, None]
Xin = np.concatenate([X44, xnum, c4, xcat], axis=1)
ci = [names.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
c4_index = X44.shape[1] + xnum.shape[1]
cat_idx = ci + [c4_index] + list(range(c4_index + 1, Xin.shape[1]))
mask = np.ones(len(y), bool)
Xn, Xc, cards = G.prep(Xin, mask, cat_idx)
G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)

sl = np.arange(0, len(y), 30, dtype=np.int64)
np.save(OUT / "ref_xin.npy", Xin[sl].astype(np.float64))
np.save(OUT / "ref_rows.npy", sl)

for seed in (42, 1, 777):
    p = torch.load(OUT / f"msa_seed{seed}.pt", map_location="cpu",
                   weights_only=False)
    m = M.make_model(Xn.shape[1], cards, seed)
    m.load_state_dict(p["model_state"])
    m = m.to(M.DEVICE)
    d, s = M.predict(m, sl)
    np.save(OUT / f"ref_direct_s{seed}.npy", d)
    print(f"  seed {seed}  direct 평균 {d.mean():.5f}", flush=True)
    del m
    torch.cuda.empty_cache()
print("  덤프 완료", len(sl), "행")
