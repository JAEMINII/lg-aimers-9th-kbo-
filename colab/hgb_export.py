# -*- coding: utf-8 -*-
"""배치용 HistGB 를 학습해 순수 numpy(npz)로 내보낸다.

제출본 의존성은 numpy 와 pandas 뿐이다. sklearn 피클을 실으면 평가 서버의
버전 불일치로 로드가 깨질 수 있어, 트리를 평탄한 배열로 내보내고 추론도
numpy 로 한다. submit_jaemin_2(958) 에서 쓰던 경로 그대로다.

구성
    학습 2019~2024 전체 (VS=2025), 시즌가중 2.0**(season-2019)
    브랜치 셋 all / futures / regular, 시드 1개
    추론에서 0.6 x all + 0.4 x (경기유형별) 로 라우팅한다 — 989 를 만든 방식이다.

    시드가 1개인 이유는 추론 시간이다. HistGB 는 이 앙상블에서 비중 0.1 이라
    시드를 늘려 얻는 것보다 트리 수가 세 배로 늘어 제한 시간을 위협하는 쪽이 크다.

피처는 features44 의 X44 다. CatBoost 와 같은 행렬을 쓰므로 추론에서
build_features 결과를 그대로 넣으면 된다.
"""
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DATA = "open (1)/data"
OUTZ = os.path.join(SC, "_dl", "hgb.npz")
VS = 2025
DECAY = 2.0
SEED = 42
BRANCHES = ("all", "futures", "regular")

import features44 as F                                          # noqa: E402

HP = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
          min_samples_leaf=500, l2_regularization=1.0, early_stopping=False)


if __name__ == "__main__":
    t0 = time.time()
    d = F.build(DATA, VS=VS)
    X, y, m_tr = d["X44"], d["y"], d["m_tr"]
    season = d["season"].astype(np.float64)
    isf = d["is_f"]
    tr_idx = np.where(m_tr)[0]
    print(f"  학습 {len(tr_idx):,}행  피처 {X.shape[1]}  시즌 "
          f"{int(season[tr_idx].min())}~{int(season[tr_idx].max())}")

    models = []
    for br in BRANCHES:
        sel = (np.ones(len(tr_idx), bool) if br == "all"
               else (isf[tr_idx] if br == "futures" else ~isf[tr_idx]))
        idx = tr_idx[sel]
        w = DECAY ** (season[idx] - 2019)
        m = HistGradientBoostingClassifier(random_state=SEED, **HP)
        m.fit(X[idx].astype(np.float64), y[idx].astype(int), sample_weight=w)
        models.append(m)
        print(f"  {br:8s} {len(idx):>10,}행  트리 {len(m._predictors)}")

    # ------------------------------------------------------------ 평탄화
    rows = []
    for m in models:
        for stage in m._predictors:
            for pr in stage:
                assert not pr.nodes["is_categorical"].any(), "범주형 분기는 못 내보낸다"
                rows.append(pr.nodes)
    T = len(rows)
    M = max(len(r) for r in rows)
    per = T // len(models)
    assert T % len(models) == 0, "모델마다 트리 수가 같아야 한다"

    feat = np.zeros((T, M), np.int32)
    thr = np.zeros((T, M), np.float64)
    left = np.zeros((T, M), np.int64)
    right = np.zeros((T, M), np.int64)
    leaf = np.ones((T, M), np.uint8)
    mgl = np.zeros((T, M), np.uint8)
    val = np.zeros((T, M), np.float64)
    for i, nd in enumerate(rows):
        k = len(nd)
        feat[i, :k] = nd["feature_idx"].astype(np.int32)
        thr[i, :k] = nd["num_threshold"]
        # 트리 내부 지역 인덱스를 전역 인덱스로 미리 바꿔 둔다. 추론에서
        # (T, M) 을 1차원으로 펴서 순회하기 때문이다.
        left[i, :k] = nd["left"].astype(np.int32) + i * M
        right[i, :k] = nd["right"].astype(np.int32) + i * M
        leaf[i, :k] = nd["is_leaf"]
        mgl[i, :k] = nd["missing_go_to_left"]
        val[i, :k] = nd["value"]
    max_depth = int(max(int(nd["depth"].max()) for nd in rows)) + 2
    base = np.array([float(np.ravel(m._baseline_prediction)[0]) for m in models],
                    np.float64)

    os.makedirs(os.path.dirname(OUTZ), exist_ok=True)
    np.savez_compressed(
        OUTZ, feat=feat, thr=thr, left=left, right=right, leaf=leaf, mgl=mgl,
        val=val, baselines=base, max_depth=np.int64(max_depth),
        n_models=np.int64(len(models)), per=np.int64(per),
        branches=np.array(BRANCHES), features=np.array(list(d["F44"])),
        decay=np.float64(DECAY), vs=np.int64(VS), seed=np.int64(SEED))
    print(f"  트리 {T}개  최대노드 {M}  깊이 {max_depth}  "
          f"{os.path.getsize(OUTZ)/1e6:.1f}MB  {time.time()-t0:.0f}s")

    # ------------------------------------------------------- 변환 검증
    # npz 순회가 sklearn 예측과 같은 값을 내는지 확인한다. 다르면 조용히
    # 틀린 제출본이 나간다.
    sys.path.insert(0, os.path.join(SC, ".."))
    chk = np.random.default_rng(0).choice(len(X), 3000, replace=False)
    Xc = X[chk].astype(np.float64)
    z = np.load(OUTZ, allow_pickle=False)
    from hgb_infer import predict_branches                      # noqa: E402
    got = predict_branches(Xc, z)
    for a, (br, m) in enumerate(zip(BRANCHES, models)):
        want = m.predict_proba(Xc)[:, 1]
        dif = np.abs(got[:, a] - want).max()
        print(f"  검증 {br:8s} 최대차 {dif:.3e}  {'통과' if dif < 1e-9 else '실패'}")
        assert dif < 1e-9, f"{br} 변환 불일치"
