# -*- coding: utf-8 -*-
"""LUPI 교사의 OOF 소프트 타깃을 만든다 — 증류의 원료.

교사 상한 실측 (VS=2024, 매칭행): base 869.7 -> phys 1167.8 (+298).
현재 투구 물리량은 결과의 원인이라 당연히 크게 오른다. 증류의 질문은
이 중 몇 %가 물리량 없는 학생에게 옮겨지는가다.

왜 OOF 인가
    교사가 자기 학습행을 예측하면 과적합 잡음까지 학생에게 옮긴다.
    5-폴드로 나눠 각 행의 소프트 타깃을 그 행을 안 본 교사가 만든다.

만드는 것 (폴드별 — 학생 관문과 학습 구간이 같아야 한다)
    lupi_soft_2024.csv.gz   row_id, p_t   (교사들: <2024 매칭행에서 5-폴드)
    lupi_soft_2022.csv.gz   row_id, p_t   (교사들: <2022 매칭행에서 5-폴드)
    규칙 4 무관 — 전부 학습 데이터 안의 일이고 추론에는 등장하지 않는다.
    FAQ 공식 답변으로 허용 확인 (2026.08.20 DACON.GM, 질문 1~3 모두 가능).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "colab"))
DATA = os.environ.get("AIMERS_DATA", os.path.join(ROOT, "open (1)", "data"))
DL = os.environ.get("AIMERS_DL", os.path.join(ROOT, "colab", "_dl"))
MATCH = os.environ.get("LUPI_MATCH", "lupi_match.csv.gz")
FOLDS = tuple(int(x) for x in os.environ.get("LO_FOLDS", "2024,2022").split(","))
TAG = os.environ.get("LUPI_SOFT_TAG", "")
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]
KFOLD = 5
SEED = 42

import features44 as F                                          # noqa: E402
from catboost import CatBoostClassifier                         # noqa: E402

d = F.build(DATA, VS=2024, return_frame=True)
fr = d["frame"]
X44 = d["X44"].astype(np.float32)
season = fr["season"].to_numpy()
y = fr["control_success"].to_numpy(np.float64)
rid = fr["row_id"].to_numpy()

M = pd.read_csv(os.path.join(DL, MATCH), encoding="utf-8-sig")
M["ptg"] = M["pitch_type_group"].map(
    {"fastball": 0, "breaking": 1, "offspeed": 2}).fillna(3)
pos = pd.Series(np.arange(len(fr)), index=rid)
mrow = pos.reindex(M["row_id"].to_numpy()).to_numpy()
ok = np.isfinite(mrow)
M, mrow = M[ok], mrow[ok].astype(int)
P = np.full((len(fr), len(PHYS) + 1), np.nan, np.float32)
P[mrow, :len(PHYS)] = M[PHYS].to_numpy(np.float32)
P[mrow, len(PHYS)] = M["ptg"].to_numpy(np.float32)
has = np.zeros(len(fr), bool)
has[mrow] = True
XT = np.concatenate([X44, P], 1)
print(f"  매칭 {int(has.sum()):,}행")

hp = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
          verbose=0, allow_writing_files=False)
for VS in FOLDS:
    m = has & (season < VS)
    idx = np.where(m)[0]
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, KFOLD, len(idx))
    soft = np.full(len(idx), np.nan)
    t0 = time.time()
    for k in range(KFOLD):
        tr_i = idx[fold != k]
        te_i = idx[fold == k]
        mo = CatBoostClassifier(random_seed=SEED + k, **hp)
        mo.fit(XT[tr_i], y[tr_i])
        soft[fold == k] = mo.predict_proba(XT[te_i])[:, 1]
        print(f"  VS={VS} fold {k+1}/{KFOLD}  {time.time()-t0:.0f}s", flush=True)
    out = pd.DataFrame({"row_id": rid[idx], "p_t": soft})
    stem = f"lupi_soft_{TAG}_{VS}" if TAG else f"lupi_soft_{VS}"
    f = os.path.join(DL, stem + ".csv.gz")
    out.to_csv(f, index=False, encoding="utf-8-sig")
    # OOF 교사의 질 — 이게 학생이 배울 원료의 질이다
    sc = F.best_shift(soft, y[idx])[0]
    print(f"  VS={VS}  {len(idx):,}행  OOF 교사 (인샘플 아님) {sc:.1f}  저장 {f}",
          flush=True)
print("  끝")
