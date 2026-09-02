# -*- coding: utf-8 -*-
"""체제 처리를 브랜치별로 다르게 건다. 1군 손실을 없애는 게 목적이다.

앞선 실험 (VS=2024, 시드 4개, 세 브랜치에 같은 처리)
```
            전체     1군      퓨처스        base 대비
base       864.1   871.7    481.4
drop       864.7   865.6    533.4      +0.6 / -6.1 / +52.0
w0.1       869.3   865.8    569.6      +5.2 / -5.9 / +88.2
flag       869.6   869.2    547.7      +5.5 / -2.5 / +66.3
flag+w0.1  871.6   866.8    589.3      +7.5 / -4.9 /+107.9
```
퓨처스는 크게 오르는데 1군이 매번 내려간다. all 브랜치에서 퓨처스 표본을 누르면
1군 예측도 같이 나빠지기 때문이다. flag 는 표본을 안 버려서 손실이 절반이다.

그래서 브랜치마다 다르게 건다
    all       flag 만        표본을 유지해 1군을 지킨다
    futures   flag + w0.1    퓨처스만 학습하니 눌러도 1군에 영향이 없다
    regular   flag 만        원래 퓨처스를 안 쓰므로 가중은 무의미

    비교 대상으로 all 에도 약한 가중(w0.5)을 걸어보는 판본을 같이 잰다.

옛 체제 퓨처스는 학습 1,221,585행 중 105,308행(8.6%)이고
퓨처스 학습분 130,994행의 80.4% 다.
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
SEEDS = (42, 1, 777, 2)
OLD_F_MAX = 2022

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)


def sc(p, m=None):
    m = FULL if m is None else m
    return F.best_shift(p[m], yv[m])[0]


def run(seed, X, ci, wmap):
    """wmap: 브랜치 -> 옛 체제 퓨처스에 걸 가중 (1.0 이면 그대로)."""
    Xn, Xc, cards = G.prep(X, G.m_tr, ci)
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    tr_idx = np.where(G.m_tr)[0]
    old_f = G.is_f[tr_idx] & (G.season[tr_idx] <= OLD_F_MAX)
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("futures", G.is_f[tr_idx]), ("regular", ~G.is_f[tr_idx])):
        idx = tr_idx[sel]
        ow = wmap.get(br, 1.0)
        w = None if ow == 1.0 else np.where(old_f[sel], ow, 1.0).astype(np.float64)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model()
        G.train(m, idx, 2, 2e-3, w=w, seed=seed, tag=f"{br} S1")
        s2 = G.season[idx] == VS - 1
        # Stage2 는 마지막 시즌만 쓴다. 옛 체제 퓨처스(<=2022)는 거기 없다.
        G.train(m, idx[s2], 1, 2e-4, params=G.stage2_params(m), seed=seed,
                tag=f"{br} S2")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    return np.where(is_f, 0.4 * P["futures"] + 0.6 * P["all"],
                    0.4 * P["regular"] + 0.6 * P["all"])


if __name__ == "__main__":
    d = F.build(DATA, VS=VS)
    base = d["X44"]
    ci = list(d["cat_idx"])
    season, isf = d["season"].astype(np.float64), d["is_f"]
    # 새 체제인가. 퓨처스는 2023 부터, 1군은 이 데이터 범위에 전환이 없다.
    flag = ((isf & (season > OLD_F_MAX)) | (~isf)).astype(np.float32)[:, None]
    Xf = np.concatenate([base, flag], 1)
    ci_f = ci + [base.shape[1]]

    plans = [
        ("base",          base, ci,   {}),
        ("flag+w0.1(전체)", Xf,  ci_f, {"all": 0.1, "futures": 0.1, "regular": 0.1}),
        ("flag+F만눌러",     Xf,  ci_f, {"futures": 0.1}),
        ("flag+all0.5",    Xf,  ci_f, {"all": 0.5, "futures": 0.1}),
        ("flag+all0.3",    Xf,  ci_f, {"all": 0.3, "futures": 0.1}),
    ]
    res = {}
    G.log("")
    G.log("  구성              전체      1군      퓨처스    개별시드(전체)")
    for name, X, c, wmap in plans:
        t0 = time.time()
        rs = [run(sd, X, c, wmap) for sd in SEEDS]
        res[name] = rs
        r = np.mean(rs, 0)
        np.save(os.path.join(OUT, f"rg2_{name.replace('+','_').replace('(','').replace(')','')}.npy"), r)
        G.log(f"  {name:16s} {sc(r):8.1f} {sc(r, ~is_f):8.1f} {sc(r, is_f):9.1f}   "
              f"[{', '.join(f'{sc(x):.0f}' for x in rs)}]   {time.time()-t0:.0f}s")

    G.log("")
    G.log("  base 대비 짝차이")
    for name in [p[0] for p in plans[1:]]:
        row = []
        for tag, m in (("전체", None), ("1군", ~is_f), ("퓨처스", is_f)):
            dif = [sc(a, m) - sc(b, m) for a, b in zip(res[name], res["base"])]
            mu = float(np.mean(dif))
            se = float(np.std(dif, ddof=1)) / np.sqrt(len(dif))
            row.append(f"{tag} {mu:+7.1f}±{se:4.1f} {sum(1 for x in dif if x>0)}/4")
        G.log(f"    {name:16s} " + "   ".join(row))
    G.log("")
    G.log("  1군 손실이 줄면서 전체가 오르는 구성이 목표다.")
