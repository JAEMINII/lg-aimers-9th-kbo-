# -*- coding: utf-8 -*-
"""제출 script.py 의 구성요소별 추론 시간을 따로 잰다.

왜 필요한가
    TabM 시드를 늘리면 10분 제한을 넘는다. 무엇을 빼서 자리를 만들지 정하려면
    각 부분이 실제로 몇 초를 쓰는지 알아야 한다. 지금까지는 추정치뿐이었다.

TabM 은 all 패스와 분기 패스를 따로 잰다. 시드를 늘릴 때
'all 만 다중시드' 가 가능한지가 여기서 갈린다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SUB = os.environ["ENS_SUB"]
ROOT = os.environ["ENS_ROOT"]
N = int(os.environ.get("TIME_N", "60000"))
sys.path.insert(0, SUB)
os.chdir(SUB)

import script as S                                              # noqa: E402
import preprocess as PPF                                        # noqa: E402
import json                                                     # noqa: E402


def clock(label, fn):
    t = time.time()
    out = fn()
    el = time.time() - t
    print(f"  {label:26s} {el:7.1f}s   253,507행 환산 {el*253507/N/60:5.2f}분")
    return out, el


if __name__ == "__main__":
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), encoding="utf-8-sig")
    te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"), encoding="utf-8-sig")
    test = tr.sample(n=N, random_state=0).reindex(columns=list(te.columns))
    test["row_id"] = [f"TEST_{i:06d}" for i in range(len(test))]
    print(f"{N:,}행 기준\n")

    (hist, z, shift), _ = clock("모델 로드", lambda: S.load_model())
    X, _ = clock("피처 (우리 경로)", lambda: S.build_features(test, hist))
    Xv = X.values.astype(np.float64)

    _, t_cb = clock("CatBoost", lambda: S.predict_numpy(Xv, z))

    st = np.load(S.resolve("model/prep_vs2025.npz"), allow_pickle=False)
    Xn, Xc = S._fm_prep(Xv, st)
    arcs = [np.load(S.resolve(f"model/vs2025_seed{s_}.npz"), allow_pickle=False)
            for s_ in (42, 1, 777)]
    _, t_mlp = clock("flatMLP (시드3)", lambda: S._fm_predict(Xn, Xc, arcs))

    with open(S.resolve("model/history.json"), encoding="utf-8") as f:
        hist_f = PPF.deserialize_history(json.load(f))
    ordered = PPF.sort_by_row_id(test)
    Xf, t_pf = clock("피처 (지인 경로)",
                     lambda: PPF.build_inference_features(ordered, hist_f))

    meta_a, arc_a = S.load_bundle(S.resolve("model/all_tabm_seed_42.npz"))
    _, t_all = clock("TabM all 패스",
                     lambda: S.predict_frame(Xf, meta_a, arc_a, chunk_size=4096))

    gt = ordered["game_type"].astype(str).to_numpy()

    def branch_pass():
        out = np.zeros(len(ordered), dtype="float32")
        for br, code in (("futures", "F"), ("regular", "R")):
            m = gt == code
            if not m.any():
                continue
            mt, ar = S.load_bundle(S.resolve(f"model/{br}_tabm_seed_42.npz"))
            out[m] = S.predict_frame(Xf.iloc[np.flatnonzero(m)], mt, ar,
                                     chunk_size=4096)
        return out

    _, t_br = clock("TabM 분기 패스", branch_pass)

    scale = 253507 / N / 60
    print(f"\n  TabM 한 시드 합계   {(t_all+t_br)*scale:.2f}분")
    print(f"  그 밖의 전부        {(t_cb+t_mlp+t_pf)*scale:.2f}분  (피처 생성 제외분 별도)")
    print(f"\n  시드를 늘리면 (all 패스 {t_all*scale:.2f}분, 분기 패스 {t_br*scale:.2f}분)")
    for k in (2, 3):
        both = (t_all * k + t_br * k) * scale
        only = (t_all * k + t_br) * scale
        print(f"    시드 {k}   전부 다중시드 {both:.2f}분   all 만 다중시드 {only:.2f}분")
    print("\n  ※ 로컬 기준. 서버는 예전 실측으로 약 0.37배 (TabM 단독 9.4분 -> 3.5분).")
