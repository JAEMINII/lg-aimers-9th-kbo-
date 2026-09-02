# -*- coding: utf-8 -*-
"""41이 통한 (rate, dev, n) 세 쌍 형태를 안 해본 축으로 옮겨 CatBoost 에서 잰다.

왜 이 형태인가
    41은 투수x카운트12 를 편차 한 열이 아니라 **세 쌍**으로 싣는다.
        rate = (s + A*prior) / (n + A)        그 칸의 수축 비율
        dev  = rate - base                     그 투수 전체와의 차이
        n    = log1p(n)                        그 칸을 얼마나 믿을지
    n 이 있어야 트리가 "표본이 적으면 dev 를 무시" 를 분기로 배운다.
    내가 어제 죽였던 스크린은 dev 한 열만 선형 편상관으로 봤다. 그래서 못 봤다.
    (success x cnt 편상관 -0.00114 인데 실제 리더보드 +7 이었다)

왜 CatBoost 인가
    TabM 은 pitcher_id / batter_id 임베딩이 있어 선수별 구조를 이미 갖고 있다.
    CatBoost 는 두 id 를 숫자로 받아 못 배운다. 같은 피처가 한쪽엔 중복,
    한쪽엔 새 정보다. 41이 그걸 증명했다.

안 해본 칸
    bt_c12    타자 x 카운트12       CatBoost 엔 batter_id 임베딩도 없다.
                                    타자 쪽은 전체 비율 2열뿐이라 통째로 비어 있다
    pc_base   투수 x 주자상황       편차 한 열로 묶어서만 기각됐다 (-10)
    pc_inn    투수 x 이닝           아무도 CatBoost 에서 안 봤다

규율
    하나씩 넣는다. 묶지 않는다.
    VS=2024 로 먼저 거르고, 살아남은 것만 VS=2022 로 확인한다.
    판정선은 두 폴드 같은 부호 & 3/3.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "colab"))
sys.path.insert(0, os.path.join(ROOT, "nn_experiments"))
os.environ.setdefault("AIMERS_ROOT", ROOT)
DATA = os.path.join(ROOT, "open (1)", "data")
VS = int(os.environ.get("TA_VS", "2024"))
SEEDS = (1, 42, 777)
ALPHA = 200.0
DECAY = 2.0
TARGET = "control_success"

import features44 as F                                          # noqa: E402
import train_c12_submit as T                                    # noqa: E402
from catboost import CatBoostClassifier                         # noqa: E402


def add_triple(d, ent, keyvals, tag):
    """41의 add_c12 와 같은 규약. 시즌 cumsum -> shift(1) 로 자기 시즌은 안 본다.

    ent      선수 열 이름 (pitcher_id / batter_id)
    keyvals  그 행의 축 값 (정수 배열)
    """
    z = pd.DataFrame({"e": d[ent].to_numpy(), "season": d["season"].to_numpy(),
                      "k": keyvals, "y": d[TARGET].to_numpy(np.float64)})
    b = z.groupby(["e", "season", "k"], sort=False)["y"].agg(["size", "sum"]) \
         .unstack(fill_value=0.0)
    b.columns = [f"{x}_{int(k)}" for x, k in b.columns]
    b = b.reindex(sorted(b.columns), axis=1)
    keys = np.array(sorted({int(c.rsplit("_", 1)[1]) for c in b.columns}), np.int64)
    cc = b.groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0.0)
    idx = pd.MultiIndex.from_arrays([z["e"].to_numpy(), z["season"].to_numpy()])
    sz = cc[[f"size_{int(k)}" for k in keys]].reindex(idx).fillna(0.0).to_numpy()
    sm = cc[[f"sum_{int(k)}" for k in keys]].reindex(idx).fillna(0.0).to_numpy()
    pos = np.clip(np.searchsorted(keys, keyvals), 0, len(keys) - 1)
    hit = keys[pos] == keyvals
    rr = np.arange(len(z))
    n = np.where(hit, sz[rr, pos], 0.0)
    s = np.where(hit, sm[rr, pos], 0.0)
    g = float(z["y"].mean())
    prior = z.groupby("k")["y"].mean().reindex(keys).fillna(g).to_numpy()[pos]
    n_all, s_all = sz.sum(1), sm.sum(1)
    base = (s_all + ALPHA * g) / (n_all + ALPHA)
    rate = (s + ALPHA * prior) / (n + ALPHA)
    return {f"{tag}_rate": rate.astype("float32"),
            f"{tag}_dev": (rate - base).astype("float32"),
            f"{tag}_n": np.log1p(n).astype("float32")}


if __name__ == "__main__":
    t0 = time.time()
    built = F.build(DATA, VS=VS, return_frame=True)
    fr = built["frame"]
    base_f = list(built["F44"])
    fr, _ = T.add_c12(fr, return_tables=True)
    fr, _ = T.add_cmh(fr, return_tables=True)
    BASE = base_f + ["pc_c12_rate", "pc_c12_dev", "pc_c12_n",
                     "pc_cmh_rate", "pc_cmh_dev", "pc_cmh_n"]

    cnt12 = fr["cnt12"].to_numpy(np.int64)
    bs = pd.factorize(fr["base_state"].astype(str))[0].astype(np.int64)
    inn = np.clip(fr["inning"].fillna(1).astype(int).to_numpy(np.int64), 1, 9)
    CAND = {
        "bt_c12": ("batter_id", cnt12),
        "pc_base": ("pitcher_id", bs),
        "pc_inn": ("pitcher_id", inn),
    }
    for tag, (ent, kv) in CAND.items():
        for c, v in add_triple(fr, ent, kv, tag).items():
            fr[c] = v
        print(f"  {tag:8s} {ent:11s} 수준 {len(np.unique(kv)):>3d}  "
              f"dev sd {fr[tag+'_dev'].std():.5f}", flush=True)

    season = fr["season"].to_numpy()
    y = fr[TARGET].to_numpy(np.float64)
    gt = fr["game_type"].to_numpy()
    tr, te = season < VS, season == VS
    w = DECAY ** (season[tr].astype(np.float64) - 2019.0)
    isf, yv = gt[te] == 1, y[te]
    print(f"\n  VS={VS}  학습 {int(tr.sum()):,}  검증 {int(te.sum()):,}  "
          f"자료 {time.time()-t0:.0f}초")

    ARMS = [("base", [])] + [(t, [f"{t}_rate", f"{t}_dev", f"{t}_n"]) for t in CAND]
    hp = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
              verbose=0, allow_writing_files=False)
    masks = {"all": np.ones(int(tr.sum()), bool),
             "regular": gt[tr] == 0, "futures": gt[tr] == 1}
    BL = {}
    for nm, extra in ARMS:
        feats = BASE + extra
        X = fr[feats].to_numpy(np.float32)
        t1 = time.time()
        pg = {}
        for g, m in masks.items():
            pg[g] = []
            for sd in SEEDS:
                mo = CatBoostClassifier(random_seed=sd, **hp)
                mo.fit(X[tr][m], y[tr][m], sample_weight=w[m])
                pg[g].append(mo.predict_proba(X[te])[:, 1])
        BL[nm] = [np.where(isf, 0.6*pg["all"][i] + 0.4*pg["futures"][i],
                           0.6*pg["all"][i] + 0.4*pg["regular"][i])
                  for i in range(len(SEEDS))]
        np.save(os.path.join(ROOT, "colab", "_dl", f"ta{VS}_{nm}.npy"),
                np.asarray(BL[nm]))
        print(f"    {nm:8s} (+{len(extra)}열) 끝 {time.time()-t1:5.0f}초", flush=True)

    sc = lambda p, m: F.best_shift(p[m], yv[m])[0]
    allm = np.ones(len(yv), bool)
    print(f"\n{'='*84}\n  (rate, dev, n) 세 쌍 — CatBoost, VS={VS}, 하나씩\n{'='*84}")
    print(f"  {'팔':9s} {'전체':>9s} {'1군':>9s} {'퓨처스':>9s}   기준선 대비")
    for nm, _ in ARMS:
        p = np.mean(BL[nm], 0)
        line = f"  {nm:9s} {sc(p,allm):9.1f} {sc(p,~isf):9.1f} {sc(p,isf):9.1f}"
        if nm != "base":
            dd = [sc(a, allm) - sc(b, allm) for a, b in zip(BL[nm], BL["base"])]
            mu = float(np.mean(dd)); se = float(np.std(dd, ddof=1))/np.sqrt(len(dd))
            line += (f"   {mu:+6.1f}+-{se:4.1f} t={mu/max(se,1e-9):5.2f} "
                     f"{sum(1 for v in dd if v>0)}/3  [{' '.join(f'{v:+.1f}' for v in dd)}]")
        print(line)
    print("\n  참고  pc_c12+pc_cmh 는 이 자리에서 리더보드 +7 이었다.")
    print("  배치 비중은 0.30(CatBoost) x 0.70(신50) = 0.21 이다. 여기 점수에 곱해서 봐라.")
