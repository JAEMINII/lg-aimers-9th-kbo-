# -*- coding: utf-8 -*-
"""submit_18(1048) 구성 위에서 한 번에 하나씩만 바꿔 잰다.

왜 지인 전처리 위에서 재나
    submit_19 가 1020 으로 떨어졌을 때 네 가지를 한꺼번에 바꿨다. 그중
    전처리 교체는 리더보드에서 한 번도 이득이 확인된 적이 없다.
        지인 전처리   submit_15 1046   submit_18 1048
        우리 전처리   submit_16 1044   submit_17 1041   submit_19 1020
    관문은 우리 쪽이 +6.7 이라 했지만 전처리는 사실상 다른 계열이고,
    관문은 계열 간 순위를 못 매긴다는 걸 이미 세 번 확인했다.
    그래서 리더보드가 가장 높았던 파이프라인을 고정해 두고 그 위에서 잰다.

무엇을 재나 (전부 submit_18 과 같은 3브랜치 0.6:0.4 확률 혼합)
    base        아무것도 안 함 = submit_18 그대로
    regime      abs_regime 플래그 + 옛 체제 퓨처스(<=2022) 학습가중 0.1
    agata       저중요도 40% 열만 증강 (NeurIPS 2024 AGATa 적응판)
    both        둘 다

채점 규약
    시드마다 따로 점수를 내고 시드끼리 짝지어 차이를 본다.
    시드평균 위에서 재면 안 된다 — branch_post 에서 4시드 평균 위에 얹어
    재는 바람에 regular 브랜치를 잘못 뺐고 그게 submit_19 실패의 한 축이다.
    배치는 1시드이므로 부품도 1시드에서 재야 한다.

    최적 시프트(판별력)와 고정 시프트(배치에서 실제 받는 값)를 같이 찍는다.

    폴드는 VS=2024 하나뿐이다. 체제 처리는 2022(검증연도가 옛 체제)와
    2023(학습 퓨처스가 전부 옛 체제)에서 구조가 달라 그 두 폴드가 무의미하다.
    그래서 이 결과는 한 폴드짜리다. 그 한계를 안고 읽어야 한다.
"""
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
SEEDS = (42, 1, 777, 2)
OLD_F_MAX = 2022
FIXED = -0.005                    # submit_18 이 실제로 쓰는 시프트
AUG_FRAC = 0.40                   # AGATa 의 하위 40%
AUG_P = 1.0                       # 배치마다 증강 (AGATa 는 epoch 마다)

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402
from train_chan_3 import preprocess as PP                       # noqa: E402

VS = int(os.environ.get("VS", 2024))
gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)


def sc(p, m=None):
    m = FULL if m is None else m
    return F.best_shift(p[m], yv[m])[0]


def scf(p, m=None):
    m = FULL if m is None else m
    return F.bss(F.shift(p[m], FIXED), yv[m])


# ------------------------------------------------------------------ 지인 전처리
def friend_matrix():
    tr = PP.sort_by_row_id(pd.read_csv(os.path.join(DATA, "train.csv"),
                                       encoding="utf-8-sig"))
    hist = PP.fit_history_tables(tr[tr.season < VS])
    X = PP.transform_features(tr, hist, train_mode=True)
    return X.to_numpy(dtype=np.float32), tr["row_id"].to_numpy(), list(X.columns)


# ------------------------------------------------------------------ AGATa
class Aug:
    """저중요도 열만 masking / shuffling / cutmix 로 변형한다.

    AGATa 는 Transformer self-attention 으로 열 중요도를 얻는다. TabM 은
    attention 이 없으므로 같은 자리에 CatBoost 중요도를 쓴다. 중요도는
    학습 구간(2019~2023)에서만 쟀다 — 검증 연도를 보면 규칙 4 위반이고
    증강 대상 선정이 검증에 맞춰지면 관문 점수를 못 믿는다.

    중요한 열을 건드리지 않는 것이 이 기법의 요지다. 무작위 마스킹이
    표 데이터에서 성능을 떨어뜨리는 이유가 핵심 상호작용을 깨기 때문이라
    저중요도만 고른다.
    """

    def __init__(self, nsel, csel):
        self.nsel = nsel          # Xn 안에서의 열 번호
        self.csel = csel          # Xc 안에서의 열 번호

    def __call__(self, xn, xc, g):
        if AUG_P < 1.0 and torch.rand(1, generator=g).item() > AUG_P:
            return xn, xc
        mode = int(torch.randint(0, 3, (1,), generator=g).item())
        b = xn.shape[0]
        perm = torch.randperm(b, generator=g).to(xn.device)
        xn, xc = xn.clone(), xc.clone()
        if mode == 0:                                   # masking
            if len(self.nsel):
                xn[:, self.nsel] = 0.0                  # 표준화 후 평균
            if len(self.csel):
                xc[:, self.csel] = 0                    # prep 의 미지 코드
        elif mode == 1:                                 # shuffling
            if len(self.nsel):
                xn[:, self.nsel] = xn[perm][:, self.nsel]
            if len(self.csel):
                xc[:, self.csel] = xc[perm][:, self.csel]
        else:                                           # cutmix
            if len(self.nsel):
                lam = float(torch.rand(1, generator=g).item())
                xn[:, self.nsel] = (lam * xn[:, self.nsel]
                                    + (1 - lam) * xn[perm][:, self.nsel])
            if len(self.csel):                          # 범주는 가중합이 없다
                sw = (torch.rand(b, len(self.csel), generator=g) < 0.5
                      ).to(xc.device)
                xc[:, self.csel] = torch.where(sw, xc[perm][:, self.csel],
                                               xc[:, self.csel])
        return xn, xc


def train_aug(model, idx, epochs, lr, aug, params=None, w=None, bs=2048,
              wd=3e-4, clip=5.0, seed=42, tag=""):
    """G.train 과 같되 배치에 증강을 건다. aug=None 이면 G.train 과 동일."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 9973)
    ps = params if params is not None else [p for p in model.parameters()
                                            if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=wd)
    nstep = max(epochs * int(np.ceil(len(idx) / bs)), 1)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=nstep)
    ii = torch.from_numpy(idx)
    ww = None if w is None else torch.from_numpy(w.astype(np.float32))
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(idx))
        tot = 0.0
        for s in range(0, len(idx), bs):
            b = ii[perm[s:s + bs]]
            xn = G.XN[b].to(G.DEV, non_blocking=True)
            xc = G.XC[b].to(G.DEV, non_blocking=True)
            yb = G.YY[b].to(G.DEV, non_blocking=True)
            if aug is not None:
                xn, xc = aug(xn, xc, g)
            opt.zero_grad(set_to_none=True)
            lg = model(xn, xc)
            if ww is None:
                loss = G.loss_fn(lg, yb)
            else:
                t = yb[:, None, None].expand_as(lg)
                per = (0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
                            lg, t, reduction="none")
                       + 0.5 * (lg.sigmoid() - t).square()).mean(dim=(1, 2))
                wb = ww[perm[s:s + bs]].to(G.DEV)
                loss = (per * wb).sum() / wb.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps, clip)
            opt.step()
            sch.step()
            tot += float(loss) * len(b)
        G.log(f"      {tag} epoch {ep+1}/{epochs}  loss {tot/len(idx):.6f}")
    return model


# ------------------------------------------------------------------ 한 판
def run(seed, X, ci, wmap, aug_names, cols):
    """3브랜치를 학습해 submit_18 과 같은 0.6:0.4 확률 혼합으로 돌려준다."""
    Xn, Xc, cards = G.prep(X, G.m_tr, ci)
    G.Xn, G.cards = Xn, cards
    G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)

    aug = None
    if aug_names:
        ciset = set(ci)
        ni = [j for j in range(X.shape[1]) if j not in ciset]
        npos = {c: a for a, c in enumerate(ni)}        # 원열 -> Xn 열
        cpos = {c: a for a, c in enumerate(ci)}        # 원열 -> Xc 열
        sel = [cols.index(nm) for nm in aug_names if nm in cols]
        nsel = torch.tensor([npos[j] for j in sel if j in npos],
                            dtype=torch.long, device=G.DEV)
        csel = torch.tensor([cpos[j] for j in sel if j in cpos],
                            dtype=torch.long, device=G.DEV)
        aug = Aug(nsel, csel)
        G.log(f"      증강: 수치 {len(nsel)}열  범주 {len(csel)}열")

    tr_idx = np.where(G.m_tr)[0]
    old_f = G.is_f[tr_idx] & (G.season[tr_idx] <= OLD_F_MAX)
    P = {}
    for br, m in (("all", np.ones(len(tr_idx), bool)),
                  ("futures", G.is_f[tr_idx]), ("regular", ~G.is_f[tr_idx])):
        idx = tr_idx[m]
        ow = wmap.get(br, 1.0)
        w = None if ow == 1.0 else np.where(old_f[m], ow, 1.0).astype(np.float64)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        mdl = G.make_model()
        train_aug(mdl, idx, 2, 2e-3, aug, w=w, seed=seed, tag=f"{br} S1")
        s2 = G.season[idx] == VS - 1
        # Stage2 는 증강 없이. 마지막 시즌에 수준을 맞추는 단계라
        # 여기서 입력을 흔들면 보정이 흐려진다.
        train_aug(mdl, idx[s2], 1, 2e-4, None, params=G.stage2_params(mdl),
                  seed=seed, tag=f"{br} S2")
        P[br] = G.predict(mdl, gate)
        del mdl
        torch.cuda.empty_cache()
    return np.where(is_f, 0.4 * P["futures"] + 0.6 * P["all"],
                    0.4 * P["regular"] + 0.6 * P["all"])


if __name__ == "__main__":
    Xf, order_f, cols = friend_matrix()
    tr_raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
    pos = pd.Series(np.arange(len(order_f)), index=order_f)
    Xf = Xf[pos.reindex(tr_raw["row_id"].to_numpy()).to_numpy()]
    ci = [cols.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]

    imp = np.load(os.path.join(SC, "featimp.npy"))
    k = int(round(AUG_FRAC * len(cols)))
    low = np.argsort(-imp)[-k:]
    aug_names = [cols[j] for j in sorted(low)]

    season, isf = G.season.astype(np.float64), G.is_f
    flag = ((isf & (season > OLD_F_MAX)) | (~isf)).astype(np.float32)[:, None]
    Xr = np.concatenate([Xf, flag], 1)
    cols_r = cols + ["abs_regime"]
    ci_r = ci + [Xf.shape[1]]
    W10 = {"all": 0.1, "futures": 0.1, "regular": 0.1}

    G.log(f"\n  지인 전처리 {Xf.shape}   범주 {len(ci)}열")
    G.log(f"  AGATa 증강 대상 {k}개 / {len(cols)}개")
    G.log(f"    {', '.join(aug_names)}\n")

    plans = [("base",   Xf, ci,   cols,   {},   None),
             ("regime", Xr, ci_r, cols_r, W10,  None),
             ("agata",  Xf, ci,   cols,   {},   aug_names),
             ("both",   Xr, ci_r, cols_r, W10,  aug_names)]
    res = {}
    G.log("  구성      시드별 단독(최적시프트)                     평균     고정시프트")
    for name, X, c, cl, wm, an in plans:
        t0 = time.time()
        ps = [run(sd, X, c, wm, an, cl) for sd in SEEDS]
        res[name] = ps
        np.save(os.path.join(OUT, f"fb_{name}.npy"), np.mean(ps, 0))
        for i, sd in enumerate(SEEDS):
            np.save(os.path.join(OUT, f"fb_{name}_s{sd}.npy"), ps[i])
        solos = [sc(p) for p in ps]
        fx = [scf(p) for p in ps]
        G.log(f"  {name:8s} [{', '.join(f'{v:7.1f}' for v in solos)}]  "
              f"{np.mean(solos):8.1f}  {np.mean(fx):8.1f}   {time.time()-t0:.0f}s")

    G.log("\n  base 대비 짝차이 (시드끼리 짝. 배치가 1시드이므로 1시드에서 잰다)")
    for name in [p[0] for p in plans[1:]]:
        row = []
        for tag, f_, m in (("최적", sc, None), ("고정", scf, None),
                           ("1군", sc, ~is_f), ("퓨처스", sc, is_f)):
            dif = [f_(a, m) - f_(b, m) for a, b in zip(res[name], res["base"])]
            mu = float(np.mean(dif))
            se = float(np.std(dif, ddof=1)) / np.sqrt(len(dif))
            t = mu / se if se > 1e-9 else 0.0
            row.append(f"{tag} {mu:+7.1f}+-{se:4.1f} t={t:5.2f} "
                       f"{sum(1 for x in dif if x > 0)}/{len(dif)}")
        G.log(f"    {name:8s} " + "  ".join(row))

    G.log("\n  한 폴드다. t 가 커도 연도는 하나뿐이라는 걸 잊지 말 것.")
    G.log("  제출 전에는 이전 제출본과 같은 입력에서 예측평균 차이를 재고")
    G.log("  시프트를 -4 x 차이 만큼 옮길 것. submit_19 는 그걸 안 해서 10~20점을 잃었다.")
