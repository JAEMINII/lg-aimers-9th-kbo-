# -*- coding: utf-8 -*-
"""배치된 CatBoost 가 시즌가중 2.0 으로 학습됐는지 지문으로 확인한다.

왜 확인이 필요한가
    trees.npz 에는 트리만 들어 있고 학습 가중치는 안 들어간다. config.json 은
    TabM 만 기록한다. 그래서 패키지만 보고는 알 수 없다.
    기억 파일과 실험 스크립트 7개가 전부 DECAY=2.0 이라 그렇게 믿고 있었을 뿐이다.

방법 — 지문 대조
    같은 피처·같은 하이퍼로 decay 를 바꿔가며 학습해서, 배치본 예측과
    상관이 가장 높은 것을 찾는다. 일반화 성능을 보는 게 아니라 **어떤 조리법으로
    만들어졌나** 를 보는 것이므로 인샘플로 재도 된다.

    배치본 30모델은 [전체 x10, R단독 x10, F단독 x10] 이다. 후보는 전체 데이터로
    학습하므로 **전체 그룹 10개 평균**과 비교해야 조건이 같다.

잡음 바닥
    같은 decay, 다른 시드끼리의 상관이 잡음 바닥이다. 후보 간 차이가 그보다
    작으면 못 가른다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, os.path.dirname(SC))
DATA = "open (1)/data"
PKG = "submit_30"
ITERS, DEPTH, LR, L2 = 400, 4, 0.05, 1.0

from catboost import CatBoostClassifier                         # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402
import importlib.util                                            # noqa: E402

spec = importlib.util.spec_from_file_location(
    "pkgscript", os.path.join(PKG, "script.py"))
S = importlib.util.module_from_spec(spec)
S.__dict__["__name__"] = "pkgscript"
try:
    spec.loader.exec_module(S)
except SystemExit:
    pass

tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                   encoding="utf-8-sig"))
y = tr[PP.TARGET].to_numpy(int)
season = tr["season"].to_numpy(np.float64)
os.chdir(PKG)
hist, z, shift = S.load_model()
X = S.build_features(tr.drop(columns=[PP.TARGET]), hist)
os.chdir("..")
Xv = X.to_numpy(dtype=np.float64)
print(f"  피처 {Xv.shape}   결측 {int(np.isnan(Xv).sum()):,}")

# ---- 배치본의 '전체' 그룹 10개만 떼어낸다
n_models = int(z["n_models"])
n_grp = int(z["n_per_group"]) if "n_per_group" in z.files else n_models
per = z["cb_leaf"].shape[0] // n_models
T_all = n_grp * per
print(f"  배치본 {n_models}모델 x {per}트리   전체그룹 {n_grp}개 -> {T_all}트리")


class _Z(dict):
    @property
    def files(self):
        return list(self.keys())


zz = _Z({k: z[k] for k in z.files})
zz["cb_leaf"] = z["cb_leaf"][:T_all]
zz["cb_feat"] = z["cb_feat"][:T_all]
zz["cb_thr"] = z["cb_thr"][:T_all]
zz["sp_idx"] = z["sp_idx"][:T_all]
zz["n_models"] = np.int32(n_grp)
zz["gt_col"] = np.int32(-1)          # 라우팅 끄고 단순 평균
zz.pop("n_per_group", None)
os.chdir(PKG)
p_dep = np.asarray(S.predict_numpy(Xv, zz, chunk=8192), np.float64).ravel()
os.chdir("..")
print(f"  배치본 전체그룹  평균 {p_dep.mean():.5f}  SD {p_dep.std():.5f}\n")


def fit(decay, seed):
    w = decay ** (season - 2019.0)
    m = CatBoostClassifier(iterations=ITERS, learning_rate=LR, depth=DEPTH,
                           l2_leaf_reg=L2, verbose=0, random_seed=seed,
                           allow_writing_files=False, thread_count=6)
    t0 = time.time()
    m.fit(Xv, y, sample_weight=w)
    p = m.predict_proba(Xv)[:, 1].astype(np.float64)
    return p, time.time() - t0


CAND = [(1.0, 42), (1.5, 42), (2.0, 42), (3.0, 42), (2.0, 7)]
res = {}
for dc, sd in CAND:
    p, el = fit(dc, sd)
    res[(dc, sd)] = p
    r = float(np.corrcoef(p, p_dep)[0, 1])
    print(f"  decay {dc:<4} seed {sd:<3} {el:5.0f}s  평균 {p.mean():.5f}  "
          f"배치본과 상관 {r:.5f}")

floor = float(np.corrcoef(res[(2.0, 42)], res[(2.0, 7)])[0, 1])
print(f"\n  잡음 바닥 (decay 2.0, 시드 42 vs 7)  {floor:.5f}")
rank = sorted({dc for dc, _ in CAND},
              key=lambda d: -float(np.corrcoef(res[(d, 42)], p_dep)[0, 1]))
print(f"  상관 순위  {rank}")
best = rank[0]
r1 = float(np.corrcoef(res[(best, 42)], p_dep)[0, 1])
r2 = float(np.corrcoef(res[(rank[1], 42)], p_dep)[0, 1])
print(f"\n  1위 decay {best} ({r1:.5f})  2위 decay {rank[1]} ({r2:.5f})  "
      f"격차 {r1-r2:+.5f}")
print("  " + (f"decay {best} 이 배치본이다. 격차가 잡음보다 크다."
               if (r1 - r2) > (1 - floor) * 0.5 else
               "격차가 잡음 수준이라 못 가른다."))
