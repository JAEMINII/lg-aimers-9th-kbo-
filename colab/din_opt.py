# -*- coding: utf-8 -*-
"""DIN 최적화 스윕 — Stage2 / 시즌감쇠 / 에폭. 관문이 믿을 수 있는 부류다.

왜
    DIN 은 TabM 하이퍼파라미터를 그대로 물려받아 2에폭 돌린 무튜닝 상태로
    리더보드 +6 을 냈다. TabM 을 끌어올린 장치들이 DIN 엔 하나도 없다.
        Stage2      TabM 은 마지막 시즌 1에폭 미세조정으로 시즌가중을 대체한다
        시즌감쇠     CatBoost 는 2.0 감쇠가 996->1021 이었다
                    (단 TabM 은 감쇠를 얹으면 단조 악화 0/3 — Stage2 가 우월했다)
    "관문은 최적화만 맞힌다" — 스케줄러/시즌가중 부류는 관문 판정이 리더보드와 맞았다.

팔 (전부 DIN 만 바꾼다. 기준선 혼합은 고정)
    base    2에폭 (= 배치 현행)
    s2e1    2에폭 + 마지막 시즌 1에폭 lr 2e-4
    s2e2    2에폭 + 마지막 시즌 2에폭
    decay2  2에폭, 가중 2.0**(season-2019) (옛퓨처스 0.1 곱 유지)
    ep4     4에폭 (코사인 T_max=4)

채점  혼합 = 0.75*기준선 + 0.25*DIN팔,  w 고정 0.25 (배치 그대로)
      팔별 이득 = 혼합(팔) - 혼합(base팔), 시드별. 판정선 두 폴드 & 3/3.
"""
import os
import sys
import time

import numpy as np
import torch
from torch import nn

SC = os.path.dirname(os.path.abspath(__file__))
for p in (SC, "/root/aimers", os.path.dirname(SC)):
    if p not in sys.path:
        sys.path.insert(0, p)
DL = os.environ.get("AIMERS_DL", "/root/aimers/_dl")
SEEDS = (42, 1, 777)
FOLDS = tuple(int(x) for x in os.environ.get("DO_FOLDS", "2022,2024").split(","))
W_DIN = 0.25

import features44 as F                                          # noqa: E402
import ctr_zoo as Z                                             # noqa: E402
from din_gate import baseline                                   # noqa: E402

ARMS = ("base", "s2e1", "s2e2", "decay2", "ep4")


def fit_arm(arm, D, idx, w, seed, gate, season, vs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = Z.DINBST(D["Xn"].shape[1], D["cards"], D["n_state"], D["n_cell"],
                 mode="din").to(Z.DEV)
    ep1 = 4 if arm == "ep4" else 2
    ww = w.copy()
    if arm == "decay2":
        ww = ww * (2.0 ** (season - 2019.0))
    opt = torch.optim.AdamW(m.parameters(), lr=Z.LR, weight_decay=Z.WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep1)
    XN = torch.from_numpy(D["Xn"]).to(Z.DEV)
    XC = torch.from_numpy(D["Xc"]).to(Z.DEV)
    Y = torch.from_numpy(D["y"]).to(Z.DEV)
    WT = torch.from_numpy(ww.astype(np.float32)).to(Z.DEV)
    SS = torch.from_numpy(D["SEQ_S"]).to(Z.DEV)
    SCq = torch.from_numpy(D["SEQ_C"]).to(Z.DEV)
    SM = torch.from_numpy(D["SEQ_M"]).to(Z.DEV)
    CU = torch.from_numpy(D["CUR"]).to(Z.DEV)

    def run_epochs(sub_idx, n_ep, lr=None, step_sched=True):
        if lr is not None:
            for g in opt.param_groups:
                g["lr"] = lr
        for _ in range(n_ep):
            m.train()
            perm = np.random.permutation(sub_idx)
            for a in range(0, len(perm), Z.BS):
                b = torch.from_numpy(perm[a:a + Z.BS]).to(Z.DEV)
                z = m(XN[b], XC[b], SS[b], SCq[b], SM[b], CU[b])
                pr = torch.sigmoid(z)
                per = 0.5 * nn.functional.binary_cross_entropy_with_logits(
                    z, Y[b], reduction="none") + 0.5 * (pr - Y[b]) ** 2
                loss = (per * WT[b]).sum() / WT[b].sum()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
            if step_sched:
                sch.step()

    run_epochs(idx, ep1)
    if arm in ("s2e1", "s2e2"):
        s2 = idx[season[idx] == vs - 1]
        if len(s2):
            run_epochs(s2, 1 if arm == "s2e1" else 2, lr=2e-4, step_sched=False)
    m.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(gate), 8192):
            b = torch.from_numpy(gate[a:a + 8192]).to(Z.DEV)
            out.append(torch.sigmoid(
                m(XN[b], XC[b], SS[b], SCq[b], SM[b], CU[b])).cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out)


if __name__ == "__main__":
    print(f"  팔 {ARMS}  시드 {SEEDS}  w={W_DIN}   판정선 두 폴드 & 3/3\n")
    for vs in FOLDS:
        Z.VS = vs
        t0 = time.time()
        D = Z.build_inputs()
        season, isf, y = D["season"], D["isf"], D["y"].astype(np.float64)
        gate = np.where(season == vs)[0]
        yv, isf_g = y[gate], isf[gate]
        tr = np.where(D["m_tr"])[0]
        w0 = np.where(isf & (season <= 2022), 0.1, 1.0)
        base, tag = baseline(vs, yv)
        sc = lambda p: F.best_shift(p, yv)[0]
        P = {}
        for arm in ARMS:
            P[arm] = []
            for sd in SEEDS:
                ps = [fit_arm(arm, D, tr[sel], w0, sd, gate, season, vs)
                      for sel in (np.ones(len(tr), bool), ~isf[tr], isf[tr])]
                P[arm].append(np.where(isf_g, 0.6*ps[0] + 0.4*ps[2],
                                       0.6*ps[0] + 0.4*ps[1]))
            np.save(f"{DL}/do_{vs}_{arm}.npy", np.asarray(P[arm]))
        ref = [sc((1 - W_DIN) * base + W_DIN * q) for q in P["base"]]
        print(f"  VS={vs}  기준선 {sc(base):.1f} ({tag})  "
              f"혼합(base팔) {np.mean(ref):.1f}   자료+학습 {time.time()-t0:.0f}s")
        print(f"    {'팔':8s} {'DIN단독':>9s} {'혼합':>9s}   {'base팔 대비 (시드별)':>26s}  {'부호':>5s}")
        for arm in ARMS:
            p = np.mean(P[arm], 0)
            mix = [sc((1 - W_DIN) * base + W_DIN * q) for q in P[arm]]
            dd = [a - b for a, b in zip(mix, ref)]
            print(f"    {arm:8s} {sc(p):9.1f} {np.mean(mix):9.1f}   "
                  + " ".join(f"{v:+7.1f}" for v in dd)
                  + f"  {sum(1 for v in dd if v > 0)}/3", flush=True)
        print()
