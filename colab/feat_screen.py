# -*- coding: utf-8 -*-
"""버려둔 피처들을 전체채점 + TabM 으로 다시 스크리닝한다.

왜 다시 하나
    지금까지의 피처 기각은 전부 '1군 채점 + CatBoost' 에서 나왔다. 둘 다 뒤에
    결함이 드러났다.
      1군 채점  퓨처스가 무너지는 걸 못 본다. ep4 를 골랐다가 980 을 받았다.
      CatBoost  기각 근거가 모델에 묶여 있다. TabM 이 쓰는 걸 트리가 못 쓸 수 있다.
    그래서 판정을 다시 받는다. 채점은 전체(R+F), 모델은 TabM ep2 3브랜치.

후보
    n_cols   asof_pitcher_n / asof_batter_n / asof_pitcher_pitchmix_n
             제공 컬럼인데 안 쓰고 있다. 특히 pitchmix_n 은 구종비율의 표본
             크기라, 지금 모델은 그 비율이 20구에서 나왔는지 2000구에서
             나왔는지를 모른다. 신뢰도 정보가 통째로 빠져 있다.
    runners  runner_on_1b/2b/3b + num_runners_on. base_state 에 이미 들어있는
             정보지만 범주 하나로 뭉쳐 있다. 신경망엔 펼친 쪽이 나을 수 있다.
    bxh      batter x pitcher_hand 평활 성공률 + log 표본수.
             예전 실험에서 44피처 520.95 -> 546.03 이었는데 안 넣고 넘어갔다.
    pxc      pitcher x count(12칸) 평활 성공률. 예전에 기각됐다.

각 후보는 기준(44열)에 더해서만 잰다. 빼는 실험은 하지 않는다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
ALPHA = 50.0

import features44 as F                                          # noqa: E402


def smoothed(tr, keys, m_tr, prior):
    """학습 구간에서만 집계한 평활 성공률과 log 표본수. 누출 없음."""
    g = tr.loc[m_tr].groupby(keys, sort=False)["control_success"]
    n, s = g.size(), g.sum()
    rate = ((s + ALPHA * prior) / (n + ALPHA)).rename("r")
    idx = pd.MultiIndex.from_frame(tr[keys]) if len(keys) > 1 else tr[keys[0]]
    return (rate.reindex(idx).to_numpy(dtype=np.float64),
            np.log1p(n.reindex(idx).fillna(0.0).to_numpy(dtype=np.float64)))


def variants(tr, m_tr):
    """이름 -> (추가 열 이름, 추가 열 값 2차원 배열, 범주형 여부)"""
    prior = float(tr.loc[m_tr, "control_success"].mean())
    out = {}

    cols = ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]
    out["n_cols"] = (cols, tr[cols].to_numpy(dtype=np.float32), [])

    cols = ["runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on"]
    out["runners"] = (cols, tr[cols].to_numpy(dtype=np.float32), cols)

    r, c = smoothed(tr, ["batter_id", "pitcher_hand"], m_tr, prior)
    out["bxh"] = (["bxh_rate", "bxh_logn"],
                  np.stack([r, c], 1).astype(np.float32), [])

    cnt = (tr["balls_before"].astype(int) * 3
           + tr["strikes_before"].astype(int)).rename("cnt12")
    tmp = tr.assign(cnt12=cnt)
    r, c = smoothed(tmp, ["pitcher_id", "cnt12"], m_tr, prior)
    out["pxc"] = (["pxc_rate", "pxc_logn"],
                  np.stack([r, c], 1).astype(np.float32), [])
    return out


if __name__ == "__main__":
    VS = int(os.environ.get("VS", 2024))
    d = F.build(DATA, VS=VS, return_frame=True)
    tr, F44 = d["frame"], d["F44"]
    m_tr = d["m_tr"]
    base = d["X44"]

    import tabm_gate_gpu as G                                    # noqa: E402
    import torch                                                 # noqa: E402

    gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
    FULL = np.ones(len(gate), bool)
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    MLPF = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    def one_seed(X, cat_idx, seed):
        """TabM ep2 3브랜치를 한 번 학습하고 라우팅한 예측을 돌려준다.

        make_model 전에 시드를 건다. tabm_gate_gpu.train 은 함수 안에서
        torch.manual_seed 를 부르는데 그때는 이미 가중치 초기화가 끝난 뒤다.
        그래서 초기화가 '앞서 무엇을 몇 번 돌렸는지' 에 좌우됐다 — 같은 시드로도
        호출 순서가 다르면 다른 모델이 나왔다(base 가 870.0 과 860.6 으로 갈렸다).
        """
        Xn, Xc, cards = G.prep(X, m_tr, cat_idx)
        # make_model 이 읽는 건 XN/XC 가 아니라 모듈 전역 Xn, cards 다. 둘 다 건다.
        G.Xn, G.cards = Xn, cards
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
        P = {}
        tr_idx = np.where(m_tr)[0]
        for br, sel in (("all", np.ones(len(tr_idx), bool)),
                        ("futures", G.is_f[tr_idx]), ("regular", ~G.is_f[tr_idx])):
            idx = tr_idx[sel]
            torch.manual_seed(seed)            # 초기화까지 덮는다
            torch.cuda.manual_seed_all(seed)
            m = G.make_model()
            G.train(m, idx, 2, 2e-3, seed=seed, tag=f"{br} S1")
            s2 = G.season[idx] == VS - 1
            G.train(m, idx[s2], 1, 2e-4, params=G.stage2_params(m), seed=seed,
                    tag=f"{br} S2")
            P[br] = G.predict(m, gate)
            del m
            torch.cuda.empty_cache()
        return np.where(is_f, 0.4 * P["futures"] + 0.6 * P["all"],
                        0.4 * P["regular"] + 0.6 * P["all"])

    def evaluate(name, X, cat_idx):
        """시드 3개로 잰다. 시드 하나는 편차가 10점이라 판정에 못 쓴다."""
        t0 = time.time()
        ps = [one_seed(X, cat_idx, sd) for sd in (42, 1, 777)]
        solos = [F.best_shift(p[FULL], yv[FULL])[0] for p in ps]
        r = np.mean(ps, 0)
        solo = F.best_shift(r[FULL], yv[FULL])[0]
        bl = F.best_shift((0.10 * CB + 0.30 * MLPF + 0.60 * r)[FULL], yv[FULL])[0]
        np.save(os.path.join(OUT, f"feat_{name}.npy"), r)
        G.log(f"  {name:10s} 열 {X.shape[1]:3d}   시드평균 단독 {solo:7.1f}   "
              f"제출비중 {bl:7.1f}   개별 "
              f"[{', '.join(f'{v:.1f}' for v in solos)}]   {time.time()-t0:.0f}s")
        return solo, bl

    G.log("\n" + "=" * 66)
    G.log("  피처 스크리닝  (전체채점, TabM ep2 3브랜치, 시드 42)")
    G.log("=" * 66)
    b = evaluate("base", base, d["cat_idx"])

    for name, (cols, vals, catcols) in variants(tr, m_tr).items():
        X = np.concatenate([base, vals], 1)
        ci = list(d["cat_idx"]) + [len(F44) + cols.index(c) for c in catcols]
        s_, bl_ = evaluate(name, X, ci)
        G.log(f"             기준 대비  단독 {s_-b[0]:+6.1f}   제출비중 {bl_-b[1]:+6.1f}\n")

    G.log("\n  시드 3개 평균으로 쟀다. 개별 시드 편차가 10점이라 3개 평균의")
    G.log("  표준오차는 대략 3점이다. +5 미만은 판정을 보류한다.")
