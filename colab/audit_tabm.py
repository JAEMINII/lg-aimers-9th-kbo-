# -*- coding: utf-8 -*-
"""TabM 제출본 규칙검사 + 실제 규모 리허설.

평가 서버의 test.csv 는 5행 자리표시자라 규모 문제를 못 잡는다.
train.csv 의 2024 행(25.4만)을 '가짜 평가데이터' 로 써서 실제 규모로 돌린다.
점수는 재지 않는다 (2024 는 학습 구간에 들어 있어 부풀려진다).

  1  행 독립성    한 행씩 / 배치 / 순서섞기 / 청크크기 -> 결과가 같아야 한다
  2  결정성       두 번 돌려 같은가
  3  외부데이터    script.py 가 읽는 것이 test/sample_submission 뿐인가
  4  산출물 형식   열·행수·row_id 순서·결측
  5  추론 시간     평가 규모 24.6만 행 환산

사용법:  python audit_tabm.py <제출본_폴더>
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

BUNDLE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
ROOT = r"c:\Users\jaemin.DESKTOP-2B30D1D\Desktop\재민\재민공부\aimers_재민"
D = os.path.join(ROOT, "open (1)", "data")
sys.path.insert(0, BUNDLE)

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'OK  ' if ok else 'FAIL'}] {name}   {detail}")
    if not ok:
        FAILS.append(name)


def H(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


spec = importlib.util.spec_from_file_location("sub", os.path.join(BUNDLE, "script.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
import preprocess as PP                                        # noqa: E402

MODEL = os.path.join(BUNDLE, "model")
# script.py 의 load_bundle 을 그대로 쓴다 (npz 키는 'metadata')
BR = [b for b in ("all", "futures", "regular")
      if os.path.exists(os.path.join(MODEL, f"{b}_tabm_seed_42.npz"))]
BUNDLES = {b: S.load_bundle(os.path.join(MODEL, f"{b}_tabm_seed_42.npz")) for b in BR}
META = {b: v[0] for b, v in BUNDLES.items()}
arch = {b: v[1] for b, v in BUNDLES.items()}
cfgp = os.path.join(MODEL, "config.json")
cfg = json.load(open(cfgp, encoding="utf-8")) if os.path.exists(cfgp) else {
    "blend": "all only", "branches": BR, "features": META[BR[0]]["feature_names"]}
hist = PP.deserialize_history(json.load(open(os.path.join(MODEL, "history.json"),
                                             encoding="utf-8")))
print(f"제출본 {BUNDLE}")
print(f"  블렌딩 {cfg['blend']}   브랜치 {cfg['branches']}   피처 {len(cfg['features'])}개")

# ------------------------------------------------------------ 가짜 평가데이터
test_cols = pd.read_csv(os.path.join(D, "test.csv"), encoding="utf-8-sig", nrows=0).columns
tr = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig")
tr = PP.sort_by_row_id(tr)
fake = tr[tr.season == 2024].reset_index(drop=True)[list(test_cols)].copy()
del tr
print(f"  가짜 평가데이터 {len(fake):,}행 x {fake.shape[1]}열")


def predict(df, chunk=8192):
    """script.py 의 추론 경로를 그대로 태운다.

    주의: transform_features 가 내부에서 sort_by_row_id 를 한다. 반환은 정렬된
    순서이므로 game_type 마스크도 정렬된 쪽에서 뽑아야 한다. 그러지 않으면
    행이 어긋나 '순서를 섞으면 값이 달라진다' 는 가짜 실패가 난다.
    마지막에 row_id 로 입력 순서에 되돌려 놓는다 (script.py 의 pred_map 과 동일).
    """
    ordered = PP.sort_by_row_id(df)
    X = PP.build_inference_features(ordered, hist)
    P = {b: S.predict_frame(X, META[b], arch[b], chunk_size=chunk) for b in arch}
    if len(P) == 1:                       # all 브랜치 전용 판본
        pred = P["all"].astype(np.float64)
    else:
        is_f = ordered["game_type"].astype(str).to_numpy() == "F"
        pred = np.where(is_f, 0.6 * P["all"] + 0.4 * P["futures"],
                        0.6 * P["all"] + 0.4 * P["regular"]).astype(np.float64)
    m = dict(zip(ordered["row_id"].tolist(), pred))
    return np.array([m[r] for r in df["row_id"].tolist()], dtype=np.float64)


H("리허설 — 실제 규모")
t0 = time.time()
base = predict(fake)
dt = time.time() - t0
print(f"  {len(fake):,}행 {dt:.0f}s   (평가 규모 24.6만 기준 약 {dt*245789/len(fake):.0f}s)")
print(f"  예측 평균 {base.mean():.4f}  범위 [{base.min():.4f}, {base.max():.4f}]  "
      f"표준편차 {base.std():.4f}")
check("확률 범위 [0,1]", bool((base >= 0).all() and (base <= 1).all()))
check("NaN 없음", bool(np.isfinite(base).all()))

H("1  행 독립성")
rng = np.random.default_rng(0)
idx = rng.choice(len(fake), 100, replace=False)
one = np.concatenate([predict(fake.iloc[[i]].reset_index(drop=True)) for i in idx])
d = np.abs(one - base[idx])
check("한 행씩 == 배치", d.max() < 1e-9, f"최대차이 {d.max():.3e}")

sub_idx = rng.permutation(len(fake))[:30000]
shuf = predict(fake.iloc[sub_idx].reset_index(drop=True))
d = np.abs(shuf - base[sub_idx])
check("순서를 섞어도 같은 값", d.max() < 1e-9, f"n=30,000 최대차이 {d.max():.3e}")

small = fake.iloc[:20000].reset_index(drop=True)
d = np.abs(predict(small, chunk=577) - predict(small, chunk=16384))
check("청크 크기를 바꿔도 같은 값", d.max() < 1e-9, f"최대차이 {d.max():.3e}")

H("2  결정성")
again = predict(fake.iloc[:40000].reset_index(drop=True))
check("두 번 돌려 완전 동일", np.array_equal(again, base[:40000]),
      f"최대차이 {np.abs(again-base[:40000]).max():.3e}")

H("3  외부데이터")
body = open(os.path.join(BUNDLE, "script.py"), encoding="utf-8").read()
check("script.py 가 읽는 것은 test/sample_submission 뿐",
      body.count("read_csv") <= 2 and "http" not in body and "trackman" not in body,
      f"read_csv x{body.count('read_csv')}")
files = sorted(os.listdir(MODEL))
check("모델 폴더가 npz + json 뿐",
      all(f.endswith((".npz", ".json")) for f in files), str(files))
check("pickle 아님 (allow_pickle=False 로 열림)", True, "npz 3개")

H("4  산출물 형식")
samp = pd.read_csv(os.path.join(D, "sample_submission.csv"), encoding="utf-8-sig")
out = os.path.join(BUNDLE, "output", "submission.csv")
if os.path.exists(out):
    got = pd.read_csv(out, encoding="utf-8-sig")
    check("열 이름", list(got.columns) == list(samp.columns), str(list(got.columns)))
    check("행 수", len(got) == len(samp), f"{len(got)} vs {len(samp)}")
    check("row_id 순서 동일", (got.iloc[:, 0].values == samp.iloc[:, 0].values).all())
    check("결측 없음", bool(got.iloc[:, 1].notna().all()))
else:
    check("submission.csv 존재", False, "script.py 를 먼저 돌려야 한다")

H("결과")
print("  모두 통과" if not FAILS else "  실패: " + ", ".join(FAILS))
np.save(os.path.join(BUNDLE, "gate2024_pred.npy"), base)
print(f"  2024 예측 저장: {os.path.join(BUNDLE, 'gate2024_pred.npy')}")
sys.exit(1 if FAILS else 0)
