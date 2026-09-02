# -*- coding: utf-8 -*-
"""체제 처리와 등판 강도가 겹치는지 본다. 둘 다 같은 문제를 다룰 수 있다.

배경
    체제 처리 (VS=2024, 4시드 짝비교)
        flag        전체  +8.7 ± 2.6  4/4    퓨처스  +70.3
        flag+w0.1   전체 +16.7 ± 3.7  4/4    퓨처스 +140.7
    등판 강도 (시즌가중 위에 얹었을 때, 세 폴드)
        2022 +3.9    2023 +93.3    2024 -0.7

    2023 에서 유독 컸다. 그 해가 퓨처스 체제 전환 해다. 즉 등판 강도가
    '커리어 성적을 못 믿을 때 이번 시즌 표본이 얼마나 쌓였나' 로 작동했을 수 있다.
    그렇다면 체제 처리가 같은 문제를 더 직접 고치므로 둘이 겹친다.

    겹치면 하나만, 더해지면 둘 다 넣는다.

무엇을 재나 (기준은 시즌가중 없는 base — 시즌가중은 2024 전용이라 뺐다)
    base
    flag_f0.1          플래그 + futures 브랜치만 가중 0.1
    flag_f0.1+load     거기에 등판 강도 2열
    load               등판 강도만

    VS=2024. 체제 처리는 이 폴드가 배치와 같은 구조라 여기서만 타당하다
    (2022 는 검증 연도가 옛 체제, 2023 은 학습 퓨처스가 전부 옛 체제).
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


def load_cols(F44, X):
    pn = X[:, F44.index("pn_cur")].astype(np.float64)
    mo = X[:, F44.index("game_month")].astype(np.float64)
    el = np.clip(mo - 2.0, 1.0, None)
    return np.stack([np.log1p(pn) - np.log1p(el), pn / el], 1).astype(np.float32)


def run(seed, X, ci, wmap):
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
        G.train(m, idx[s2], 1, 2e-4, params=G.stage2_params(m), seed=seed,
                tag=f"{br} S2")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    return np.where(is_f, 0.4 * P["futures"] + 0.6 * P["all"],
                    0.4 * P["regular"] + 0.6 * P["all"])


if __name__ == "__main__":
    d = F.build(DATA, VS=VS)
    F44 = list(d["F44"])
    base = d["X44"]
    ci = list(d["cat_idx"])
    season, isf = d["season"].astype(np.float64), d["is_f"]
    flag = ((isf & (season > OLD_F_MAX)) | (~isf)).astype(np.float32)[:, None]
    ld = load_cols(F44, base)

    Xf = np.concatenate([base, flag], 1)                    # 44 + flag
    Xl = np.concatenate([base, ld], 1)                      # 44 + load2
    Xfl = np.concatenate([base, flag, ld], 1)               # 44 + flag + load2
    ci_f = ci + [base.shape[1]]

    plans = [("base", base, ci, {}),
             ("flag_f0.1", Xf, ci_f, {"futures": 0.1}),
             ("flag_f0.1+load", Xfl, ci_f, {"futures": 0.1}),
             ("load", Xl, ci, {})]
    res = {}
    G.log("")
    G.log("  구성              전체      1군      퓨처스    개별시드(전체)")
    for name, X, c, wmap in plans:
        t0 = time.time()
        rs = [run(sd, X, c, wmap) for sd in SEEDS]
        res[name] = rs
        r = np.mean(rs, 0)
        np.save(os.path.join(OUT, f"rl_{name.replace('+','_')}.npy"), r)
        G.log(f"  {name:16s} {sc(r):8.1f} {sc(r, ~is_f):8.1f} {sc(r, is_f):9.1f}   "
              f"[{', '.join(f'{sc(x):.0f}' for x in rs)}]   {time.time()-t0:.0f}s")

    G.log("")
    G.log("  base 대비 짝차이")
    d_ = {}
    for name in [p[0] for p in plans[1:]]:
        row = []
        for tag, m in (("전체", None), ("1군", ~is_f), ("퓨처스", is_f)):
            dif = [sc(a, m) - sc(b, m) for a, b in zip(res[name], res["base"])]
            mu = float(np.mean(dif))
            se = float(np.std(dif, ddof=1)) / np.sqrt(len(dif))
            if tag == "전체":
                d_[name] = mu
            row.append(f"{tag} {mu:+7.1f}±{se:4.1f} {sum(1 for x in dif if x>0)}/4")
        G.log(f"    {name:16s} " + "   ".join(row))

    a, b, c = d_["flag_f0.1"], d_["load"], d_["flag_f0.1+load"]
    G.log("")
    G.log(f"  따로 더하면 {a:+.1f} + {b:+.1f} = {a+b:+.1f}   같이 걸면 {c:+.1f}")
    G.log(f"  -> {'더해진다' if c > a + b - 2.0 else '겹친다'}")
