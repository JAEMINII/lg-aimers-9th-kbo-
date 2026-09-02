# -*- coding: utf-8 -*-
"""준혁 CB 재검 — 배치판 가중(0.7^역년차)으로 폴드 재학습.

저장: cbh_{VS}_{base|cand}.npy (3시드 평균, 0.6/0.4 라우팅), cbh_{VS}_jh.npy
판정은 로컬 믹스 채점에서."""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = "/root/aimers"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/colab")
sys.path.insert(0, ROOT + "/nn_experiments")
os.environ.setdefault("AIMERS_ROOT", "/root")
DATA = os.environ.get("AIMERS_DATA", ROOT + "/data")
DL = ROOT + "/colab/_dl"
import features44 as F                                          # noqa: E402
import train_c12_submit as T                                    # noqa: E402
import current_catboost_features as CF                          # noqa: E402
from catboost import CatBoostClassifier                         # noqa: E402

raw = pd.read_csv(DATA + "/train.csv", encoding="utf-8-sig")
JH = dict(iterations=230, learning_rate=0.05, l2_leaf_reg=3,
          loss_function="Logloss", random_seed=42, verbose=0,
          random_strength=1, allow_writing_files=False,
          bootstrap_type="MVS", subsample=0.8)
hp = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
          verbose=0, allow_writing_files=False, task_type="GPU", devices="0")

for VS in (2024, 2022, 2023):
    # ---- 준혁 CB 폴드 재학습 (그들 파이프라인 그대로, 무가중)
    t0 = time.time()
    hist = raw[raw.season < VS]
    prior = float(hist["control_success"].mean())
    profiles = CF.build_matchup_profiles(hist)
    tr_f = CF.transform(hist, prior, profiles)
    feats, cats = CF.feature_names(tr_f)
    tr_f = CF.prepare_categoricals(tr_f, cats)
    va_raw = raw[raw.season == VS]
    va_f = CF.prepare_categoricals(CF.transform(va_raw, prior, profiles), cats)
    wjh = 0.7 ** (float(hist["season"].max()) - hist["season"].to_numpy(np.float64))
    wjh = wjh / wjh.mean()
    mo = CatBoostClassifier(cat_features=cats, **JH)
    mo.fit(tr_f[feats], hist["control_success"].to_numpy(), sample_weight=wjh)
    jh = mo.predict_proba(va_f[feats])[:, 1]
    np.save(DL + f"/cbh_{VS}_jhw.npy", jh)
    print(f"VS{VS} 준혁CB {time.time()-t0:.0f}s  평균 {jh.mean():.4f}", flush=True)


print("jhw 끝", flush=True)
