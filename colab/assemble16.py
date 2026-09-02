# -*- coding: utf-8 -*-
"""submit_ens/model/ 을 새 TabM(우리 전처리 + 시즌가중 3.5)으로 갈아끼운다.

바뀌는 것
    들어옴   tabm_{all,futures,regular}_seed{42,1}.npz   우리 전처리, decay 3.5
    빠짐     {all,futures,regular}_tabm_seed_42.npz      지인 전처리, 가중 없음
             history.json, config.json                   지인 preprocess 룩업
             preprocess.py                               지인 피처 파이프라인

    피처 파이프라인이 하나(build_features)로 줄어든다. 파일도 셋 줄어든다.

확인하는 것
    1. 시즌가중이 실제로 걸렸는가 (meta.decay). 한 번 놓쳤던 지점이다.
    2. abs_regime 열이 들어갔는가 (meta.features 45개, 끝 열이 abs_regime).
    3. 체제 가중이 걸렸는가 (meta.regime == 'flag_w0.1').
    4. 전처리 통계가 브랜치·시드에 걸쳐 같은가.

    flatMLP 통계와의 대조는 이제 안 한다. TabM 은 46열, flatMLP 은 44열로
    입력이 달라져 통계 모양 자체가 다르다. 대신 script.py 가 meta.features 를
    보고 필요한 열을 만들어 붙이고, 안 맞으면 멈춘다.
"""
import json
import os
import shutil
import sys

import numpy as np

ROOT = r"c:\Users\jaemin.DESKTOP-2B30D1D\Desktop\재민\재민공부\aimers_재민"
DST = os.path.join(ROOT, "submit_ens", "model")
SRC = os.path.join(ROOT, "colab", "_dl")          # 서버에서 받아둔 곳
SEEDS = (42, 1, 777)
BRANCHES = ("all", "futures")   # regular 는 뺐다
EXPECT_DECAY = 0.0        # 시즌가중 없음
EXPECT_NFEAT = 45          # 44 + abs_regime
EXPECT_NCAT = 10          # 9 + abs_regime

# 옛 판본 잔여물. regular 브랜치와 flatMLP 파일도 이제 안 쓴다.
OBSOLETE = ["all_tabm_seed_42.npz", "futures_tabm_seed_42.npz",
            "regular_tabm_seed_42.npz", "history.json", "config.json",
            "tabm_regular_seed42.npz", "tabm_regular_seed1.npz",
            "prep_vs2025.npz", "vs2025_meta.json",
            "vs2025_seed42.npz", "vs2025_seed1.npz", "vs2025_seed777.npz"]


def check_meta(path):
    z = np.load(path, allow_pickle=False)
    m = json.loads(str(z["meta"].item()))
    n = os.path.basename(path)
    if abs(float(m.get("decay", 0.0)) - EXPECT_DECAY) > 1e-9:
        raise SystemExit(f"!! {n} decay={m.get('decay')} — 기대 {EXPECT_DECAY}")
    if len(m["features"]) != EXPECT_NFEAT:
        raise SystemExit(f"!! {n} 피처 {len(m['features'])}개 — 기대 {EXPECT_NFEAT}")
    if m["features"][-1] != "abs_regime":
        raise SystemExit(f"!! {n} 끝 열이 abs_regime 이 아니다: {m['features'][-1]}")
    if m.get("regime") != "flag_w0.1":
        raise SystemExit(f"!! {n} regime={m.get('regime')!r} — 기대 'flag_w0.1'")
    if len(m["cards"]) != EXPECT_NCAT:
        raise SystemExit(f"!! {n} 범주 {len(m['cards'])}개 — 기대 {EXPECT_NCAT}")
    return m, z


if __name__ == "__main__":
    missing = [f"{b}_seed{s}.npz" for s in SEEDS for b in BRANCHES
               if not os.path.exists(os.path.join(SRC, f"{b}_seed{s}.npz"))]
    if missing:
        raise SystemExit(f"받아온 파일 없음: {missing}")

    metas = []
    for s in SEEDS:
        for b in BRANCHES:
            m, z = check_meta(os.path.join(SRC, f"{b}_seed{s}.npz"))
            metas.append((b, s, m, z))
    m0 = metas[0][2]
    print(f"설정 확인  decay={m0['decay']}  ep1={m0['ep1']}  ep2={m0['ep2']}  "
          f"k={m0['k']}  vs={m0['vs']}  lowcard={m0.get('lowcard')}  "
          f"load={m0.get('load')}")
    print(f"  피처 {len(m0['features'])}개  범주 {len(m0['cards'])}개  "
          f"끝 2열 {m0['features'][-2:]}")

    # 전처리 통계가 브랜치·시드에 걸쳐 같은지 (같은 prep_stats 를 공유해야 한다)
    ref = metas[0][3]
    for b, s, _, z in metas[1:]:
        for key in ("cat_idx", "med", "mu", "sd", "has_nan"):
            if not np.array_equal(np.asarray(z[key]), np.asarray(ref[key])):
                raise SystemExit(f"!! {b}_seed{s} 의 {key} 가 다르다")
    print("브랜치·시드 간 전처리 통계 동일")

    for s in SEEDS:
        for b in BRANCHES:
            shutil.copy(os.path.join(SRC, f"{b}_seed{s}.npz"),
                        os.path.join(DST, f"tabm_{b}_seed{s}.npz"))
    print(f"복사 {len(SEEDS) * len(BRANCHES)}개")

    for f in OBSOLETE:
        p = os.path.join(DST, f)
        if os.path.exists(p):
            os.remove(p)
            print(f"  제거 {f}")
    pp = os.path.join(ROOT, "submit_ens", "preprocess.py")
    if os.path.exists(pp):
        os.remove(pp)
        print("  제거 preprocess.py")

    tot = sum(os.path.getsize(os.path.join(DST, f)) for f in os.listdir(DST))
    print(f"\nmodel/ {len(os.listdir(DST))}개 파일  {tot / 1e6:.1f} MB")
    for f in sorted(os.listdir(DST)):
        print(f"  {f:32s} {os.path.getsize(os.path.join(DST, f)) / 1e6:6.2f} MB")
