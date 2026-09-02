# -*- coding: utf-8 -*-
"""퓨처스 체제 전환을 다룬다. 시즌가중으로는 안 되는 문제다.

발견
    퓨처스 성공률이 2022 년 70.87% 에서 2023 년 47.29% 로 -23.6%p 떨어진다.
    같은 해 1군은 -0.06%p 로 거의 안 움직인다. ABS(자동 볼판정) 도입으로 보인다.
```
            R 성공률  전년대비      F 성공률  전년대비
    2021    51.28%   -1.42p       70.38%  +11.61p
    2022    50.37%   -0.91p       70.87%   +0.49p
    2023    50.31%   -0.06p       47.29%  -23.58p   <-
    2024    48.97%   -1.34p       45.93%   -1.36p
```

왜 중요한가
    학습 구간에 퓨처스가 두 체제로 섞여 있다. 2019~2022 는 옛 체제(70%),
    2023~2024 는 새 체제(47%). 예측 대상인 2025 는 새 체제다.
    즉 퓨처스 학습 데이터의 절반 이상이 다른 게임이다.

    시즌가중은 '최근일수록 맞다' 를 가정하는데 체제 전환 앞에서는 그 가정이
    깨진다. 실제로 다년 백테스트에서 시즌가중이 폴드마다 부호가 갈렸다.
```
    2022  -25.3   저드리프트 해라 옛 시즌을 버릴 이유가 없다
    2023 -230.3   옛 체제 퓨처스로 더 세게 학습해 새 체제를 크게 틀린다
    2024  +46.9   1군 드리프트가 커서 최근 쪽이 이득
```

무엇을 재나  (기준은 시즌가중 없는 base. 시즌가중과 섞으면 원인이 흐려진다)
    base        현행
    drop        퓨처스 2019~2022 행을 학습에서 제외
    w0.1        그 행들에만 가중 0.1
    flag        abs_regime 플래그를 피처로 추가 (모델이 알아서 쓰게)
    flag+w0.1   둘 다

    VS=2024 에서 잰다. 관문 학습 구간(2019~2023)에도 같은 혼재가 있어 조건이 같다.
    퓨처스 브랜치와 all 브랜치 점수를 따로 찍는다 — 효과가 어디서 나오는지 봐야 한다.
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
OLD_F_MAX = 2022          # 이 해까지의 퓨처스가 옛 체제

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)


def sc(p, m=None):
    m = FULL if m is None else m
    return F.best_shift(p[m], yv[m])[0]


def run(seed, X, ci, mode):
    """mode: base / drop / w0.1 / flag / flag+w0.1"""
    Xn, Xc, cards = G.prep(X, G.m_tr, ci)
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
    tr_idx = np.where(G.m_tr)[0]
    old_f = G.is_f[tr_idx] & (G.season[tr_idx] <= OLD_F_MAX)
    P = {}
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("futures", G.is_f[tr_idx]), ("regular", ~G.is_f[tr_idx])):
        keep = sel.copy()
        w = None
        if "drop" in mode:
            keep = keep & ~old_f
        elif "w0.1" in mode:
            w = np.where(old_f[sel], 0.1, 1.0).astype(np.float64)
        idx = tr_idx[keep]
        if len(idx) < 5000:                 # 퓨처스 drop 이면 표본이 확 준다
            P[br] = np.full(len(gate), yv.mean())
            continue
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model()
        G.train(m, idx, 2, 2e-3, w=w, seed=seed, tag=f"{br} S1")
        s2 = G.season[idx] == VS - 1
        if s2.sum() > 500:
            # Stage2 는 마지막 시즌(VS-1)만 쓴다. 옛 체제 퓨처스는 2022 이하라
            # 거기 안 들어가므로 가중을 걸 필요가 없다.
            G.train(m, idx[s2], 1, 2e-4, params=G.stage2_params(m),
                    seed=seed, tag=f"{br} S2")
        P[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()
    r = np.where(is_f, 0.4 * P["futures"] + 0.6 * P["all"],
                 0.4 * P["regular"] + 0.6 * P["all"])
    return r, P


if __name__ == "__main__":
    d = F.build(DATA, VS=VS)
    F44 = list(d["F44"])
    base = d["X44"]
    ci = list(d["cat_idx"])

    # flag: 새 체제인가. 퓨처스는 2023 부터, 1군은 이 데이터 범위에선 전환이 없다.
    season = d["season"].astype(np.float64)
    isf = d["is_f"]
    flag = ((isf & (season > OLD_F_MAX)) | (~isf)).astype(np.float32)[:, None]
    Xflag = np.concatenate([base, flag], 1)
    ci_flag = ci + [base.shape[1]]           # 범주형으로 넣는다

    tr_idx = np.where(G.m_tr)[0]
    nold = int((G.is_f[tr_idx] & (G.season[tr_idx] <= OLD_F_MAX)).sum())
    G.log(f"\n  학습 {len(tr_idx):,}행 중 옛 체제 퓨처스 {nold:,}행 "
          f"({nold/len(tr_idx)*100:.1f}%)")
    G.log(f"  퓨처스 학습분 {int(G.is_f[tr_idx].sum()):,}행 중 옛 체제가 "
          f"{nold/max(int(G.is_f[tr_idx].sum()),1)*100:.1f}%")

    plans = [("base", base, ci, ""), ("drop", base, ci, "drop"),
             ("w0.1", base, ci, "w0.1"), ("flag", Xflag, ci_flag, ""),
             ("flag+w0.1", Xflag, ci_flag, "w0.1")]
    res = {}
    G.log("")
    G.log("  구성        전체      1군       퓨처스     개별시드(전체)")
    for name, X, c, mode in plans:
        t0 = time.time()
        rs = [run(sd, X, c, mode)[0] for sd in SEEDS]
        res[name] = rs
        r = np.mean(rs, 0)
        np.save(os.path.join(OUT, f"rg{VS}_{name.replace('+','_')}.npy"), r)
        G.log(f"  {name:11s} {sc(r):8.1f} {sc(r, ~is_f):8.1f} {sc(r, is_f):9.1f}   "
              f"[{', '.join(f'{sc(x):.0f}' for x in rs)}]   {time.time()-t0:.0f}s")

    G.log("")
    G.log("  base 대비 짝차이")
    for name in ("drop", "w0.1", "flag", "flag+w0.1"):
        for tag, m in (("전체", None), ("퓨처스", is_f)):
            dif = [sc(a, m) - sc(b, m) for a, b in zip(res[name], res["base"])]
            mu = float(np.mean(dif))
            se = float(np.std(dif, ddof=1)) / np.sqrt(len(dif))
            G.log(f"    {name:11s} {tag:6s} {mu:+8.1f} ± {se:5.1f}  "
                  f"{sum(1 for x in dif if x > 0)}/{len(dif)}")
