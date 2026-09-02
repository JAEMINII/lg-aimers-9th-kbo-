# -*- coding: utf-8 -*-
"""지인 학습 코드와 우리 학습 코드의 격차가 어디서 오는지 찾는다.

왜 이게 최우선인가
    리더보드가 학습 코드로 깨끗하게 갈린다.
        지인 학습이 1군(88%)을 맡음   1046 / 1048 / 1057
        우리 학습이 1군을 맡음        1020 / 1024 / 1041 / 1044
    submit_22 는 지인 **전처리**를 쓰고도 1024 였다. 갈리는 축은 전처리가 아니다.
    이 격차를 못 메우면 우리가 찾는 개선(reg4 관문 +11.3 등)을 88% 구간에
    실을 수가 없다. 어제 그래서 -33 이 났다.

코드를 읽어 찾은 차이 두 가지 (설정값은 완전히 같다)
    1) 코사인 스케줄러 주기
         지인  CosineAnnealingLR(T_max=epochs=2), step() 을 **에폭마다**
               -> 1에폭 내내 lr=0.002, 2에폭 내내 lr=0.001
         우리  CosineAnnealingLR(T_max=에폭x배치수), step() 을 **배치마다**
               -> 0.002 에서 0 까지 감쇠, 후반부가 거의 0
    2) Stage2 학습 범위
         지인  --fine-tune-scope 기본값 "head" -> 출력층만
         우리  output + backbone.blocks[-1][0] -> 마지막 블록까지

    참고로 조기종료는 둘 다 안 걸린다. 지인 스크립트가 X_val 을 안 넘겨서
    best_state 가 None 으로 남고 복원이 일어나지 않는다. 처음엔 그게
    원인인 줄 알았는데 아니었다.

무엇을 재나 (지인 전처리, VS=2024, all 브랜치, 1군 행 채점)
    ours    현행 우리 코드
    sched   스케줄러만 지인식(에폭당 1스텝)
    head    Stage2 만 지인식(출력층만)
    both    둘 다 지인식
    friend  지인 코드를 그대로 호출  <- 기준선

    all 브랜치 하나만 돌린다. 1군 행이 88% 이고 branch_redo 에서 1군은
    all 단독이 최선이었다. 비용이 1/3 로 준다.

    전처리 통계는 양쪽 다 학습 구간에서만 뽑는다. 지인 스크립트는 X_all 로
    fit 하는데, 배치(VS=2025)에서는 X_all 이 곧 학습 구간이라 같다.
    관문에서 그대로 두면 검증 연도가 새어 비교가 더러워진다.
"""
import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, "/workspace/aimers")
OUT = "/workspace/aimers/out"
DATA = "/workspace/aimers/data"
SEEDS = (42, 1, 777)

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
R = ~is_f


def sc(p, m):
    return F.best_shift(p[m], yv[m])[0]


def train_arm(model, idx, epochs, lr, per_epoch_sched, params=None, seed=42,
              bs=2048, wd=3e-4, clip=5.0, tag=""):
    """G.train 과 같되 스케줄러 주기를 고를 수 있다."""
    torch.manual_seed(seed)
    ps = params if params is not None else [p for p in model.parameters()
                                            if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=wd)
    if per_epoch_sched:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        nstep = max(epochs * int(np.ceil(len(idx) / bs)), 1)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=nstep)
    ii = torch.from_numpy(idx)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(idx))
        tot = 0.0
        for s in range(0, len(idx), bs):
            b = ii[perm[s:s + bs]]
            xn = G.XN[b].to(G.DEV, non_blocking=True)
            xc = G.XC[b].to(G.DEV, non_blocking=True)
            yb = G.YY[b].to(G.DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = G.loss_fn(model(xn, xc), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps, clip)
            opt.step()
            if not per_epoch_sched:
                sch.step()
            tot += float(loss.detach()) * len(b)
        if per_epoch_sched:
            sch.step()
        G.log(f"      {tag} ep{ep+1}/{epochs} loss {tot/len(idx):.6f} "
              f"lr {opt.param_groups[0]['lr']:.6f}")
    return model


def stage2_params(model, scope):
    out = list(model.output.parameters())
    if scope == "last_block":
        out += list(model.backbone.blocks[-1][0].parameters())
    return out


def run_ours(seed, idx, per_epoch_sched, scope, season):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = G.make_model()
    train_arm(m, idx, 2, 2e-3, per_epoch_sched, seed=seed, tag="S1")
    s2 = season[idx] == VS - 1
    for p in m.parameters():
        p.requires_grad_(False)
    ps = stage2_params(m, scope)
    for p in ps:
        p.requires_grad_(True)
    train_arm(m, idx[s2], 1, 2e-4, per_epoch_sched, params=ps, seed=seed, tag="S2")
    for p in m.parameters():
        p.requires_grad_(True)
    p_out = G.predict(m, gate)
    del m
    torch.cuda.empty_cache()
    return p_out


def run_friend(seed, Xdf, ydf, prep, season, mask):
    """지인 코드를 그대로 호출한다. 이게 기준선이다."""
    from train_chan_3.official_tabm import OfficialTabMEstimator, TabMConfig
    from train_chan_3.train_conditional import fine_tune
    import json
    with open(os.path.join("/workspace/aimers/train_chan_3",
                           "selected_config.json"), encoding="utf-8") as f:
        sel = json.load(f)
    params = dict(sel["model"]["params"])
    params.update(loss=sel["model"].get("loss", "bce_brier"), epochs=2,
                  patience=3, device="cuda")
    est = OfficialTabMEstimator(TabMConfig.from_dict(params), seed=seed)
    est.fit(Xdf.loc[mask].reset_index(drop=True),
            ydf.loc[mask].reset_index(drop=True),
            preprocessor=copy.deepcopy(prep), verbose=False)
    rec = mask & (season == VS - 1)
    fine_tune(est, Xdf.loc[rec].reset_index(drop=True),
              ydf.loc[rec].reset_index(drop=True),
              epochs=1, learning_rate=2e-4, scope="head", seed=seed)
    return est.predict(Xdf.iloc[gate].reset_index(drop=True)).astype(np.float64)


if __name__ == "__main__":
    from train_chan_3.official_tabm import TabularPreprocessor

    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr[tr.season < VS])
    Xdf_sorted = PP.transform_features(tr, hist, train_mode=True)
    cols = list(Xdf_sorted.columns)
    # G 의 행 순서(features44)로 되돌린다
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
    pos = pd.Series(np.arange(len(tr)), index=tr["row_id"].to_numpy())
    order = pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()
    Xdf = Xdf_sorted.iloc[order].reset_index(drop=True)
    Xf = Xdf.to_numpy(dtype=np.float32)
    ydf = pd.Series(G.y.astype(np.float32))
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]

    m_tr = G.m_tr
    season = G.season.astype(np.int64)
    tr_idx = np.where(m_tr)[0]
    G.log(f"\n  지인 피처 {Xf.shape}   학습 {m_tr.sum():,}   "
          f"관문 {len(gate):,} (1군 {R.sum():,})")

    # 우리 경로용 전처리
    Xn, Xc, cards = G.prep(Xf, m_tr, ci)
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)

    # 지인 경로용 전처리 — 학습 구간에서만 fit 한다
    fprep = TabularPreprocessor(PP.TABM_CATEGORICAL_FEATURES, True)
    fprep.fit(Xdf.loc[m_tr].reset_index(drop=True))

    arms = [("ours", False, "last_block"), ("sched", True, "last_block"),
            ("head", False, "head"), ("both", True, "head")]
    P = {}
    for nm, pes, scope in arms:
        t0 = time.time()
        for s in SEEDS:
            P[(nm, s)] = run_ours(s, tr_idx, pes, scope, season)
        G.log(f"  {nm:7s} {time.time()-t0:.0f}s")
    t0 = time.time()
    for s in SEEDS:
        P[("friend", s)] = run_friend(s, Xdf, ydf, fprep, season, m_tr)
    G.log(f"  friend  {time.time()-t0:.0f}s")

    for k, v in P.items():
        np.save(os.path.join(OUT, f"cg_{k[0]}_s{k[1]}.npy"), v)

    G.log("\n  all 브랜치 단독, 최적 시프트, 시드별 계산 후 평균")
    G.log(f"  {'구성':8s} {'1군 행':>9s} {'전체':>9s} {'퓨처스':>9s}   시드별(1군)")
    base = None
    for nm in ("ours", "sched", "head", "both", "friend"):
        v = [sc(P[(nm, s)], R) for s in SEEDS]
        if base is None:
            base = v
        d = [a - b for a, b in zip(v, base)]
        G.log(f"  {nm:8s} {np.mean(v):9.1f} "
              f"{np.mean([sc(P[(nm,s)], np.ones(len(yv),bool)) for s in SEEDS]):9.1f} "
              f"{np.mean([sc(P[(nm,s)], is_f) for s in SEEDS]):9.1f}   "
              f"[{', '.join(f'{x:.1f}' for x in v)}]")

    G.log("\n  ours 대비 짝차이 (1군 행)")
    for nm in ("sched", "head", "both", "friend"):
        d = [sc(P[(nm, s)], R) - sc(P[("ours", s)], R) for s in SEEDS]
        mu = float(np.mean(d))
        se = float(np.std(d, ddof=1)) / np.sqrt(len(d))
        G.log(f"    {nm:8s} {mu:+7.1f}+-{se:5.1f} t={mu/max(se,1e-9):5.2f} "
              f"{sum(1 for x in d if x > 0)}/{len(d)}  "
              f"[{', '.join(f'{x:+.1f}' for x in d)}]")

    G.log("\n  friend 가 ours 보다 크게 높고 both 가 friend 에 가까우면 원인을 찾은 것이다.")
    G.log("  both 가 안 따라잡으면 차이가 다른 데 있다 — 그때는 전처리 세부나")
    G.log("  DataLoader 셔플 방식을 봐야 한다.")
