# -*- coding: utf-8 -*-
"""배치가 모델에 먹이는 행렬 vs 학습이 만든 행렬을 **열 단위로** 대조한다.

왜
    submit_34/35 의 이득(+7)은 전부 "검증한 구성 != 실은 구성" 을 고쳐서 나왔다.
    체제 열이 관문에선 4단계인데 배치에선 이진(또는 없음)이었다.
    그런데 그건 **이름이 같아서** 기존 검증을 통과했다 —
        if want[-1] != "abs_regime": raise   # 이름만 본다
    45번째 열 하나를 그렇게 놓쳤다. 나머지 44열은 같은 방식으로 안 봤다.

무엇을 대조하나
    학습 경로   PP.transform_features(sorted_raw, hist, train_mode=True)
                + plat_dev 를 features44 의 as-of 값으로 교체 (train_platfix 가 하는 것)
    추론 경로   PPF.build_inference_features(ordered, hist)
                (submit_35/preprocess.py — 학습에 쓴 것과 바이트 동일)

    **두 함수가 다르다.** 같은 history 를 줘도 같은 값을 낸다는 보장이 없다.

    열마다 본다
        최대차          부동소수 오차인지 진짜 다른지
        후보 집합       범주형은 값 집합이 같아야 한다 (c4 를 놓친 자리)
        결측 개수

주의
    학습은 plat_dev 를 as-of 로 **교체**하고 추론은 얼린 값을 쓴다. 이건 의도된
    것이다 — 두 정의 모두 "그 행의 시즌보다 앞선 전부" 라 규약이 같다. 그래서
    이 열의 차이는 결함이 아니다. 나머지 열에서 차이가 나오면 그게 새 발견이다.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "colab"))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "open (1)", "data")
PKG = "submit_35"
N = 30000

import features44 as F                                          # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

raw = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                    encoding="utf-8-sig"))
rng = np.random.default_rng(0)
pick = np.sort(rng.choice(len(raw), N, replace=False))
sample = raw.iloc[pick].reset_index(drop=True)
print(f"  표본 {N:,}행 (train.csv, row_id 정렬)")

# ---------------------------------------------------------------- 학습 경로
hist = PP.fit_history_tables(raw)                 # VS=2025: 전부가 학습 구간
A = PP.transform_features(raw, hist, train_mode=True).iloc[pick].reset_index(drop=True)
d = F.build(DATA, VS=2025, return_frame=True)
j = list(d["F44"]).index("plat_dev")
s = pd.Series(d["X44"][:, j].astype(np.float64),
              index=d["frame"]["row_id"].to_numpy())
A_plat = s.reindex(sample["row_id"].to_numpy()).to_numpy()
print(f"  학습 경로  {A.shape}   (plat_dev 는 as-of 로 교체됨)")

# ---------------------------------------------------------------- 추론 경로
cwd = os.getcwd()
os.chdir(os.path.join(ROOT, PKG))
sys.path.insert(0, os.getcwd())
import preprocess as PPF                                        # noqa: E402
with open("model/history.json", encoding="utf-8") as f:
    hist_pkg = PPF.deserialize_history(json.load(f))
test_like = sample.drop(columns=[PP.TARGET])
B = PPF.build_inference_features(PPF.sort_by_row_id(test_like), hist_pkg)
sys.path.pop(0)
os.chdir(cwd)
print(f"  추론 경로  {B.shape}   (패키지 history.json 사용)")

# ---------------------------------------------------------------- 대조
ca, cb = list(A.columns), list(B.columns)
if ca != cb:
    only_a = [c for c in ca if c not in cb]
    only_b = [c for c in cb if c not in ca]
    print(f"\n  **열 구성이 다르다**  학습에만 {only_a}   추론에만 {only_b}")
print(f"\n  {'열':26s} {'최대차':>12s} {'다른 행':>9s} {'결측 A/B':>12s}  비고")
bad = []
for c in ca:
    if c not in cb:
        continue
    a = A[c].to_numpy(np.float64)
    b = B[c].to_numpy(np.float64)
    na, nb = int(np.isnan(a).sum()), int(np.isnan(b).sum())
    both = np.isnan(a) & np.isnan(b)
    diff = np.where(both, 0.0, np.abs(np.nan_to_num(a) - np.nan_to_num(b)))
    mx, nd = float(diff.max()), int((diff > 1e-9).sum())
    note = ""
    if c in PP.TABM_CATEGORICAL_FEATURES:
        ua = set(np.unique(a[~np.isnan(a)]).tolist())
        ub = set(np.unique(b[~np.isnan(b)]).tolist())
        if ua != ub:
            note = f"후보집합 다름 A-B={sorted(ua-ub)[:3]} B-A={sorted(ub-ua)[:3]}"
    if nd or note or na != nb:
        bad.append(c)
        print(f"  {c:26s} {mx:12.4e} {nd:9,} {na:5,}/{nb:<6,}  {note}")
if not bad:
    print("  (44열 전부 동일)")
print(f"\n  차이 있는 열 {len(bad)}개 / {len(ca)}개")

print("\n  plat_dev 는 따로 본다 (학습은 as-of 교체, 추론은 얼린 값 — 의도된 것)")
pa, pb = A_plat, B["plat_dev"].to_numpy(np.float64)
print(f"    상관 {np.corrcoef(pa, pb)[0,1]:.4f}   최대차 {np.abs(pa-pb).max():.4e}")
print("    두 정의 모두 '그 행의 시즌보다 앞선 전부' 라 규약은 같다.")

print("\n  45번째 열(체제)은 양쪽 다 코드에서 직접 만든다")
season = sample["season"].to_numpy()
isf = sample["game_type"].astype(str).to_numpy() == "F"
old = season <= 2022
c4_train = np.where(old & isf, 0.0, np.where(old & ~isf, 1.0,
                                             np.where(isf, 2.0, 3.0)))
c4_infer = np.where(isf, 2.0, 3.0)     # 2025 는 전부 '새'
print(f"    학습(2019~2024 혼재)  후보 {sorted(set(c4_train.tolist()))}")
print(f"    추론(2025 만)         후보 {sorted(set(c4_infer.tolist()))}")
print("    추론에 옛 체제 행이 없으므로 0/1 칸은 안 쓰인다. 정상이다.")
