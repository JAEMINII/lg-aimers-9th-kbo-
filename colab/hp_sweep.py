# -*- coding: utf-8 -*-
"""TabM 하이퍼파라미터를 믿을 수 있는 관문에서 다시 훑는다.

왜 다시 하나
    기존 설정 결정(ep3/ep4/ep5/ep6/ep8/ep16, k=64, 시즌가중, 라우팅 비중)은 전부
    단일 실행 비교였다. 그런데 tabm_gate_gpu.train 은 torch.manual_seed 를 함수
    '안'에서 부르므로 가중치 초기화가 시드를 안 받았다. 같은 코드·같은 시드로
    base 가 870.0 과 860.6 으로 갈렸다.

    실제 단일 시드 편차는 24점이다. 옛 표들의 차이는 대개 10점 안쪽이었으니
    그 비교들은 노이즈를 읽은 것이다. ep2 만 리더보드 실측(1041 vs ep4 980)이
    있어 근거가 있다.

이번 규약
    초기화 전에 시드를 건다.  시드 3개 평균으로만 판정한다.
    채점은 전체(R+F).  피처는 우리 전처리 (지인 것보다 시드별로 일관되게 높았다).
    3시드 평균의 표준오차가 약 3점이므로 +5 미만은 판정 보류.
"""
import os
import sys
import time

import numpy as np
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
SEEDS = (42, 1, 777)

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)
VS = int(os.environ.get("VS", 2024))

CASES = {
    "ep2":        dict(),                       # 현 설정. 리더보드 1041
    "ep3":        dict(ep1=3),
    "ep4":        dict(ep1=4),                  # 리더보드 980 이었는데 뽑기였나
    "ep2_k64":    dict(k=64),
    "ep2_nos2":   dict(stage2=False),           # Stage2 가 정말 필요한가
    "ep2_sw2":    dict(decay=2.0),              # 우리 CatBoost 를 +22 올린 장치
}


def one(seed, ep1=2, ep2=1, lr1=2e-3, lr2=2e-4, k=32, stage2=True, decay=None):
    P = {}
    tr_idx = np.where(G.m_tr)[0]
    W = None if decay is None else decay ** (G.season[tr_idx].astype(float) - 2019)
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("futures", G.is_f[tr_idx]), ("regular", ~G.is_f[tr_idx])):
        idx = tr_idx[sel]
        w = None if W is None else W[sel]
        torch.manual_seed(seed)                 # 초기화까지 덮는다
        torch.cuda.manual_seed_all(seed)
        m = G.make_model(k=k)
        G.train(m, idx, ep1, lr1, w=w, seed=seed, tag=f"{br} S1")
        if stage2:
            s2 = G.season[idx] == VS - 1
            G.train(m, idx[s2], ep2, lr2, params=G.stage2_params(m),
                    w=None if w is None else w[s2], seed=seed, tag=f"{br} S2")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    return np.where(is_f, 0.4 * P["futures"] + 0.6 * P["all"],
                    0.4 * P["regular"] + 0.6 * P["all"])


if __name__ == "__main__":
    d = F.build(DATA, VS=VS)
    Xn, Xc, cards = G.prep(d["X44"], G.m_tr, d["cat_idx"])
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    MLPF = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    G.log("\n" + "=" * 74)
    G.log("  설정        시드평균 단독   제출비중   개별 시드")
    G.log("=" * 74)
    base = None
    for name, kw in CASES.items():
        t0 = time.time()
        # k 를 바꾸면 make_model 인자가 달라진다. one() 이 받아서 넘긴다.
        ps = [one(sd, **kw) for sd in SEEDS]
        solos = [F.best_shift(p[FULL], yv[FULL])[0] for p in ps]
        r = np.mean(ps, 0)
        solo = F.best_shift(r[FULL], yv[FULL])[0]
        bl = F.best_shift((0.10 * CB + 0.30 * MLPF + 0.60 * r)[FULL], yv[FULL])[0]
        np.save(os.path.join(OUT, f"hp_{name}.npy"), r)
        if base is None:
            base = (solo, bl)
            delta = ""
        else:
            delta = f"   기준대비 {solo-base[0]:+6.1f} / {bl-base[1]:+6.1f}"
        G.log(f"  {name:11s} {solo:7.1f}      {bl:7.1f}   "
              f"[{', '.join(f'{v:.0f}' for v in solos)}]   {time.time()-t0:.0f}s{delta}")

    G.log("\n  +5 미만은 판정 보류. ep4 가 ep2 와 비슷하게 나오면")
    G.log("  리더보드 980 은 설정 탓이 아니라 뽑기였다는 뜻이 된다.")
