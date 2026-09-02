# -*- coding: utf-8 -*-
"""plat_dev 를 as-of 로 고쳐서 배치용(VS=2025) 모델을 학습한다.

발견한 결함
    지인 preprocess 의 plat_dev 는 **투수당 값 하나**를 학습창 전체로 계산한다.
        pid.map(history["plat_l"])
    그래서 2019년 학습 행이 2020~2023 결과를 본다. 학습 중 미래 누출이다.

    features44 는 (투수, 시즌)별 누적을 한 시즌 밀어낸다 (cumsum().shift(1)).
    2021년 행은 2019~2020 만 본다. as-of 다.

    44열 중 **이 한 열만** 다르고 나머지 43열은 상관 1.00000 / 최대차 0.0000 이다.

측정 (지인 피처 고정, plat_dev 만 교체, 3시드 짝비교)
        VS=2022   friend 2429.7   asof +19.5 ± 3.5  t=5.52  3/3   drop -3.7
        VS=2024   friend  906.1   asof +24.2 ± 3.0  t=8.19  3/3   drop -7.8
    drop 이 음수라 열을 지우면 안 된다 — 누출된 채로도 유용한 열이었다.
    as-of 로 **교체**가 정답이다.

추론 경로는 안 건드린다
    2025 시험 행에게는 '2019~2024 전부' 가 정당한 과거다. 지금 고정 테이블이
    이미 그 값을 준다. 즉 학습 행만 고치면 되고 preprocess.py 는 그대로 둔다.
    제출 패키지의 추론 코드가 안 바뀌므로 위험이 작다.

바꾸는 것은 하나뿐
    submit_27(LB 1058)과 비교해 시드(42), 배치(2048), lr(0.002),
    fine-tune-scope(last_block), 브랜치 구성 전부 그대로다.
    plat_dev 계산만 다르다.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/workspace/aimers")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "colab"))

import features44 as F                                          # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402
from train_chan_3.official_tabm import (OfficialTabMEstimator,  # noqa: E402
                                        TabMConfig,
                                        TabularPreprocessor)
from train_chan_3.train_conditional import fine_tune            # noqa: E402
from train_chan_3.build_submission import export_one            # noqa: E402

DATA = ROOT / "data"
ART = ROOT / os.environ.get("ART", "art_platfix")
NPZ = ROOT / os.environ.get("NPZ", "npz_platfix")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42").split(",")]
CFG = os.environ.get("CFG", "train_chan_3/selected_config.json")
TGT = "control_success"


def asof_plat_dev(train):
    """as-of plat_dev 를 features44 에서 **그대로 가져온다**.

    재구현하지 않는 이유 — features44 의 계산은 단순한 cumsum-shift 가 아니다.
        ALPHA_PLAT = 300 으로 평활하고,
        사전분포가 전역 평균이 아니라 **(투수손, 타자손)별 리그 값**이다.
    직접 다시 짜면 관문에서 +24.2 를 낸 그 열과 미세하게 달라질 수 있다.
    측정한 것과 다른 걸 싣는 게 제일 나쁘다. 그래서 원본을 호출한다.

    VS=2025 로 부르면 m_tr 이 전체가 되어 배치 구성과 맞는다. shift(1) 은
    그대로 걸리므로 각 행은 여전히 자기 시즌 이전만 본다.

    행 순서는 row_id 로 맞춘다 — features44 는 자체 정렬을 쓴다.
    """
    d = F.build(str(DATA), VS=2025, return_frame=True)
    j = list(d["F44"]).index("plat_dev")
    vals = d["X44"][:, j].astype(np.float64)
    rid = d["frame"]["row_id"].to_numpy()
    s = pd.Series(vals, index=rid)
    out = s.reindex(train["row_id"].to_numpy()).to_numpy(np.float64)
    if np.isnan(out).any():
        raise ValueError(f"row_id 정렬 실패: 결측 {int(np.isnan(out).sum())}개")
    return out


if __name__ == "__main__":
    cfg = json.loads((ROOT / CFG).read_text(encoding="utf-8"))
    ART.mkdir(parents=True, exist_ok=True)
    NPZ.mkdir(parents=True, exist_ok=True)

    train = PP.sort_by_row_id(pd.read_csv(DATA / "train.csv",
                                          encoding="utf-8-sig"))
    history = PP.fit_history_tables(train)          # VS=2025: 전부가 학습 구간
    X_all = PP.transform_features(train, history, train_mode=True)
    y_all = train[PP.TARGET].reset_index(drop=True)

    old = X_all["plat_dev"].to_numpy(np.float64).copy()
    new = asof_plat_dev(train)
    X_all["plat_dev"] = new
    r = np.corrcoef(old, new)[0, 1]
    print(f"  plat_dev 교체   두 판본 상관 {r:.4f}   "
          f"최대차 {np.abs(old-new).max():.4f}")
    print(f"    기존 표적상관 {np.corrcoef(old, y_all)[0,1]:+.5f}  "
          f"(누출로 부풀려진 값)")
    print(f"    신규 표적상관 {np.corrcoef(new, y_all)[0,1]:+.5f}  (as-of)\n")

    params = dict(cfg["model"]["params"])
    prep = TabularPreprocessor(
        params.get("categorical_features", PP.TABM_CATEGORICAL_FEATURES),
        params.get("add_missing_indicators", True)).fit(X_all)
    masks = {"all": np.ones(len(train), bool),
             "futures": train["game_type"].astype(str).to_numpy() == "F",
             "regular": train["game_type"].astype(str).to_numpy() == "R"}
    PP_json = PP.serialize_history(history)
    (ART / "history.json").write_text(json.dumps(PP_json, ensure_ascii=False),
                                      encoding="utf-8")

    for branch, mask in masks.items():
        Xb = X_all.loc[mask].reset_index(drop=True)
        yb = y_all.loc[mask].reset_index(drop=True)
        for sd in SEEDS:
            t0 = time.time()
            p = dict(params, loss=cfg["model"].get("loss", "brier"),
                     epochs=int(cfg["full_train_epochs"]),
                     patience=int(cfg["full_train_epochs"]) + 1, device="cuda")
            model = OfficialTabMEstimator(TabMConfig.from_dict(p), seed=int(sd))
            model.fit(Xb, yb, preprocessor=prep, verbose=False)
            # Stage2 — 마지막 시즌, last_block (배치본과 같은 설정)
            s2 = train.loc[mask, "season"].to_numpy() == 2024
            fine_tune(model, Xb.loc[s2].reset_index(drop=True),
                      yb.loc[s2].reset_index(drop=True),
                      epochs=1, learning_rate=2e-4, scope="last_block",
                      seed=int(sd))
            ck = ART / f"{branch}_s{sd}.pt"
            model.save(ck)
            dst = NPZ / f"{branch}_s{sd}.npz"
            export_one(ck, dst)
            print(f"  {branch:8s} seed {sd}  {time.time()-t0:6.0f}s  "
                  f"-> {dst.name} {dst.stat().st_size/1e6:.1f}MB")
    print(f"\n  완료. {NPZ}")
