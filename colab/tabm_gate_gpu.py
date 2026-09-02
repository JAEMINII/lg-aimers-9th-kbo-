# -*- coding: utf-8 -*-
"""TabM 관문 실험 — 지인 구조 + 우리가 알아낸 것.

지인에게서 가져온 것 (official_tabm.py 에서 확인한 정확한 API)
    TabM.make(n_num_features, cat_cardinalities, d_out=1, num_embeddings=..., ...)
    수치 임베딩  rtdl_num_embeddings.LinearReLUEmbeddings(n, d_embedding=16)
    loss        0.5 x BCE(logits) + 0.5 x Brier(sigmoid)      <- d_out=1, sigmoid
    예측        logits.sigmoid().mean(dim=1)                  <- k개 멤버 평균
    Stage2      output + backbone.blocks[-1][0] 만 풀고 lr 1/10
    k=32 / n_blocks=3 / d_block=256 / dropout=0.1 / batch 2048 / lr 2e-3 / wd 3e-4

우리가 얹는 것
    · 연도 간격 있는 관문 (2019~2023 학습 -> 2024 채점, 1군만)
      지인은 리더보드 점수만 알지 관문 점수를 모른다. 우리는 환산식이 있어
      재현이 제대로 됐는지 그 자리에서 확인할 수 있다
          리더보드 ~= 0.97 x 관문 + 146      (5회 측정, 3회 연속 적중)
          지인 TabM 1047  ->  관문 환산 929
    · 최적 시프트로 비교 — 수준 보정과 판별력이 섞이면 최적 비중이 움직인다
    · 시즌 가중치 2.0 — 우리 CatBoost 를 +22 올린 것. 지인은 안 쓴다.
      Stage2 와 같은 일(최근 시즌 강조)을 다르게 하는 것이라 겹칠 수도 있다
    · CatBoost/MLP 와의 상관 — 단독 점수가 아니라 이게 앙상블 이득을 정한다
          CatBoost <-> MLP 0.907 -> +5.9 / <-> LGB 0.962 -> +0.5

사용법
    python tabm_gate_gpu.py base            지인 설정 그대로 (재현 확인)
    python tabm_gate_gpu.py sw2             + 시즌가중 2.0
    python tabm_gate_gpu.py nostage2        Stage2 없이
    python tabm_gate_gpu.py all             전부 순차
"""
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DATA = os.environ.get("AIMERS_DATA", "/workspace/aimers/data")
OUT = os.environ.get("AIMERS_OUT", "/workspace/aimers/out")
TM_PATH = os.environ.get("TM_PATH", "")
os.makedirs(OUT, exist_ok=True)
PROG = os.path.join(OUT, "progress.txt")


def log(s):
    print(s, flush=True)
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(s + "\n")


import features44 as F                                          # noqa: E402
from rtdl_num_embeddings import LinearReLUEmbeddings            # noqa: E402
from tabm import TabM                                           # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------ 데이터
t0 = time.time()
VS = int(os.environ.get("VS", 2024))
d = F.build(DATA, VS=VS)
X, y = d["X44"], d["y"]
if TM_PATH:
    # Prior-season TrackMan 2-seam aggregates, aligned to row-id order.
    tm = pd.read_csv(TM_PATH)
    tmcols = [c for c in tm.columns if c.startswith("tm2s")]
    keys = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                       usecols=["row_id", "pitcher_id", "season"])
    keys["_r"] = keys.row_id.astype(str).str.extract(r"(\d+)$", expand=False).astype("int64")
    keys = keys.sort_values("_r").reset_index(drop=True)
    tm = tm.sort_values(["pitcher_id", "season"])
    rows = {}
    for pid, gg in tm.groupby("pitcher_id", sort=False):
        cn = {c: 0.0 for c in tmcols}; cd = {c: 0.0 for c in tmcols}
        for season, gs in gg.groupby("season", sort=True):
            rows[(int(pid), int(season))] = {
                c: (cn[c] / cd[c] if cd[c] > 0 else np.nan) for c in tmcols}
            ww = gs["tm_n"].to_numpy(np.float64)
            for c in tmcols:
                vv = gs[c].to_numpy(np.float64)
                ok = np.isfinite(vv) & np.isfinite(ww)
                if ok.any():
                    cn[c] += float(np.sum(vv[ok] * ww[ok]))
                    cd[c] += float(np.sum(ww[ok]))
    M = np.full((len(keys), len(tmcols)), np.nan, dtype=np.float32)
    for i, (pid, season) in enumerate(zip(keys.pitcher_id, keys.season)):
        q = rows.get((int(pid), int(season)))
        if q is not None:
            M[i] = [q[c] for c in tmcols]
    X = np.concatenate([X, M], axis=1)
    log(f"TrackMan 2-seam {len(tmcols)}열 추가 -> X {X.shape}")
m_tr, m_va, is_f, season = d["m_tr"], d["m_va"], d["is_f"], d["season"]
gate = np.where(m_va)[0]
mm = ~is_f[gate]
yv = y[gate].astype(np.float64)
log(f"피처 {time.time()-t0:.0f}s   X {X.shape}   학습 {m_tr.sum():,}   "
    f"관문 1군 {mm.sum():,}   실제 성공률 {yv[mm].mean():.4f}")


def prep(X, m_tr, cat_idx):
    """지인 TabularPreprocessor 와 같은 규약. 통계는 학습 구간에서만."""
    n, p = X.shape
    ci = np.asarray(cat_idx)
    ni = np.asarray([j for j in range(p) if j not in set(cat_idx)])
    Xc = np.zeros((n, len(ci)), dtype=np.int64)
    cards = []
    for a, j in enumerate(ci):
        vals = np.unique(X[m_tr, j])
        vals = vals[~np.isnan(vals)]
        pos = np.clip(np.searchsorted(vals, X[:, j]), 0, max(len(vals) - 1, 0))
        hit = (vals[pos] == X[:, j]) if len(vals) else np.zeros(n, bool)
        Xc[:, a] = np.where(hit, pos + 1, 0)
        cards.append(len(vals) + 1)
    Xn = X[:, ni].astype(np.float64)
    miss = np.isnan(Xn)
    med = np.nanmedian(Xn[m_tr], 0)
    Xn = np.where(miss, med, Xn)
    mu, sd = Xn[m_tr].mean(0), Xn[m_tr].std(0) + 1e-6
    Xn = ((Xn - mu) / sd).astype(np.float32)
    has_nan = miss[m_tr].any(0)
    if has_nan.any():
        Xn = np.concatenate([Xn, miss[:, has_nan].astype(np.float32)], 1)
    return Xn, Xc, np.asarray(cards)


Xn, Xc, cards = prep(X, m_tr, d["cat_idx"])
log(f"수치 {Xn.shape[1]}열  범주 {Xc.shape[1]}열  카디널리티 {cards.tolist()}")
XN = torch.from_numpy(Xn)
XC = torch.from_numpy(Xc)
YY = torch.from_numpy(y.astype(np.float32))

CB = np.load(os.path.join(SC, "cb_gate.npy" if VS == 2024 else f"cb_gate{VS}.npy"))
MLP = (np.load(os.path.join(SC, "mlp_gate.npy")).astype(np.float64)
       if VS == 2024 else CB.copy())   # 2023 폴드엔 MLP 가 없어 CB 로 대체
CUR = 0.8 * CB + 0.2 * MLP if VS == 2024 else CB.copy()                      # 현 제출본 = 리더보드 1021
BASE = F.best_shift(CUR[mm], yv[mm])[0]
log(f"기준  CatBoost {F.best_shift(CB[mm], yv[mm])[0]:.1f} / "
    f"MLP {F.best_shift(MLP[mm], yv[mm])[0]:.1f} / 현 제출본 {BASE:.1f} (= LB 1021)")
log(f"지인 TabM 은 LB 1012 -> 관문 환산 약 893\n")


# ------------------------------------------------------------------ 모델
def make_model(k=32, n_blocks=3, d_block=256, dropout=0.1, d_emb=16, arch="tabm"):
    return TabM.make(
        n_num_features=Xn.shape[1],
        cat_cardinalities=[int(c) for c in cards],
        d_out=1,
        num_embeddings=LinearReLUEmbeddings(Xn.shape[1], d_embedding=d_emb),
        arch_type=arch, k=k, n_blocks=n_blocks,
        d_block=d_block, dropout=dropout,
    ).to(DEV)


def loss_fn(logits, target):
    """0.5 BCE + 0.5 Brier.  지인 official_tabm._loss 와 동일."""
    t = target[:, None, None].expand_as(logits)
    brier = (logits.sigmoid() - t).square().mean()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, t)
    return 0.5 * bce + 0.5 * brier


def train(model, idx, epochs, lr, params=None, w=None, bs=2048, wd=3e-4,
          clip=5.0, seed=42, tag=""):
    """w 를 주면 시즌 가중치. 지인에겐 없고 우리 CatBoost 를 +22 올린 장치다."""
    torch.manual_seed(seed)
    ps = params if params is not None else [p for p in model.parameters()
                                            if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=wd)
    # 코사인을 **에폭 단위**로 돈다. 배치마다 돌려 T_max 를 전체 스텝수로 두면
    # lr 이 0 까지 완전히 감쇠하는데, 에폭이 2개뿐이라 후반부가 통째로 버려진다.
    #
    # codegap 실험 (지인 전처리, VS=2024, all 브랜치, 1군 행, 3시드 짝비교)
    #     배치마다 감쇠 (예전)   869.5
    #     에폭마다 감쇠 (지금)   888.8   +19.2 +- 6.7  t=2.86  3/3
    #     지인 코드 그대로       874.6
    # 이 한 줄이 '우리 학습 코드가 지인보다 나쁘다' 의 정체였다. 리더보드에서
    # 우리 학습 1020/1024/1041/1044 vs 지인 학습 1046/1048/1057 로 갈리던 축이다.
    #
    # 주의: 이 줄을 고치기 전에 나온 관문 수치는 전부 덜 학습된 모델 위의 값이다.
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ii = torch.from_numpy(idx)
    ww = None if w is None else torch.from_numpy(w.astype(np.float32))
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(idx))
        tot = 0.0
        for s in range(0, len(idx), bs):
            b = ii[perm[s:s + bs]]
            xn = XN[b].to(DEV, non_blocking=True)
            xc = XC[b].to(DEV, non_blocking=True)
            yb = YY[b].to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            lg = model(xn, xc)
            if ww is None:
                loss = loss_fn(lg, yb)
            else:                      # 가중 평균으로 바꿔 준다
                t = yb[:, None, None].expand_as(lg)
                per = (0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
                            lg, t, reduction="none")
                       + 0.5 * (lg.sigmoid() - t).square()).mean(dim=(1, 2))
                wb = ww[perm[s:s + bs]].to(DEV)
                loss = (per * wb).sum() / wb.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps, clip)
            opt.step()
            tot += float(loss) * len(b)
        sch.step()                     # 에폭 끝에서 한 번
        log(f"      {tag} epoch {ep+1}/{epochs}  loss {tot/len(idx):.6f}  "
            f"lr {opt.param_groups[0]['lr']:.6f}")
    return model


@torch.no_grad()
def predict(model, idx, bs=8192):
    model.eval()
    out = np.empty(len(idx), np.float64)
    ii = torch.from_numpy(idx)
    for s in range(0, len(idx), bs):
        b = ii[s:min(s + bs, len(idx))]
        lg = model(XN[b].to(DEV), XC[b].to(DEV))
        out[s:s + len(b)] = lg.sigmoid().mean(dim=1).squeeze(-1).double().cpu().numpy()
    return out


def stage2_params(model):
    """지인과 동일: output + backbone.blocks[-1][0]"""
    for p in model.parameters():
        p.requires_grad_(False)
    ps = list(model.output.parameters())
    ps += list(model.backbone.blocks[-1][0].parameters())
    for p in ps:
        p.requires_grad_(True)
    return ps



def route(P, w_branch):
    """분기 모델 비중 w_branch, 전체 모델 비중 1-w_branch."""
    return np.where(is_f[gate],
                    w_branch * P["futures"] + (1 - w_branch) * P["all"],
                    w_branch * P["regular"] + (1 - w_branch) * P["all"])


def sweep_route(P, name):
    """라우팅 비율을 학습 없이 훑는다. 0.0 = 전체모델만, 1.0 = 분기모델만."""
    log(f"    라우팅 비율 (분기 비중)   [{name}]")
    best = (-1e9, None)
    for wb in np.arange(0.0, 1.01, 0.1):
        p = route(P, wb)
        solo = F.best_shift(p[mm], yv[mm])[0]
        bw, bs_ = 0.0, BASE
        for wv in np.arange(0.05, 0.85, 0.05):
            s_ = F.best_shift(((1 - wv) * CUR + wv * p)[mm], yv[mm])[0]
            if s_ > bs_:
                bs_, bw = s_, wv
            lb = 1021 + 0.97 * (bs_ - BASE)
        if bs_ > best[0]:
            best = (bs_, wb)
        log(f"      분기 {wb:.1f}  단독 {solo:7.1f}  TabM비중 {bw:.2f}  "
            f"섞은뒤 {bs_:7.1f}  LB {lb:6.0f}")
    log(f"      -> 최적 분기비중 {best[1]:.1f}")


def run(name, decay=None, stage2=True, ep1=2, ep2=1, lr1=2e-3, lr2=2e-4,
        k=32, recent=None, seed=42):
    """3브랜치(all/futures/regular)를 0.6:0.4 로 합친다."""
    tr_idx = np.where(m_tr)[0]
    W = None if decay is None else decay ** (season[tr_idx].astype(np.float64) - 2019)
    P = {}
    t0 = time.time()
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("futures", is_f[tr_idx]), ("regular", ~is_f[tr_idx])):
        idx = tr_idx[sel]
        w = None if W is None else W[sel]
        m = make_model(k=k)
        train(m, idx, ep1, lr1, w=w, seed=seed, tag=f"{br} S1")
        if stage2:
            s2m = season[idx] == (VS - 1 if recent is None else recent)
            ps = stage2_params(m)
            train(m, idx[s2m], ep2, lr2, params=ps,
                  w=None if w is None else w[s2m], seed=seed, tag=f"{br} S2")
        P[br] = predict(m, gate)
        del m
        gc.collect()
        torch.cuda.empty_cache()
    # 브랜치별로 저장해두면 라우팅 비율 스윕이 학습 없이 된다
    for br_, pv in P.items():
        np.save(os.path.join(OUT, f"br{VS}_{name}_{br_}.npy"), pv)
    # 전체 모델 60% + 분기 모델 40%.  TRAINING_SPEC 및 우리 CatBoost 와 동일.
    # route() 는 분기 비중을 받으므로 0.4 다.
    # 우리 CatBoost 는 관문에서 이 비율이 평평했다 (0.4/0.5/0.6 -> 907.2/907.1/906.7).
    # TabM 도 그런지는 sweep_route 가 바로 찍어 준다.
    p = route(P, 0.4)
    np.save(os.path.join(OUT, f"tabm_{name}.npy" if VS == 2024 else f"tabm{VS}_{name}.npy"), p)
    sweep_route(P, name)

    solo = F.best_shift(p[mm], yv[mm])[0]
    cor_cb = np.corrcoef(p[mm], CB[mm])[0, 1]
    cor_ml = np.corrcoef(p[mm], MLP[mm])[0, 1]
    bw, bs_ = 0.0, BASE
    for wv in np.arange(0.05, 0.85, 0.05):
        s = F.best_shift(((1 - wv) * CUR + wv * p)[mm], yv[mm])[0]
        if s > bs_:
            bs_, bw = s, wv
    log(f"\n  [{name}]  {time.time()-t0:.0f}s")
    log(f"    단독 {solo:.1f}   (지인 재현 목표 929)")
    log(f"    상관  CatBoost {cor_cb:.4f}   MLP {cor_ml:.4f}")
    log(f"    현 제출본에 비중 {bw:.2f} 로 얹으면 {bs_:.1f}  ({bs_-BASE:+.1f})"
        f"  -> LB 예상 {1021 + 0.97*(bs_-BASE):.0f}\n")
    return dict(name=name, solo=solo, cor_cb=cor_cb, cor_mlp=cor_ml,
                w=bw, blended=bs_, lb=1021 + 0.97 * (bs_ - BASE))


CASES = {
    "base":      dict(),                       # 지인 설정 그대로
    "sw2":       dict(decay=2.0),              # + 우리 시즌가중
    "nostage2":  dict(stage2=False),           # Stage2 가 정말 핵심인지
    "sw2_nos2":  dict(decay=2.0, stage2=False),
    "ep4":       dict(ep1=4),
    # epoch 을 늘리자 상관이 0.9723 -> 0.9336 으로 떨어지고 이득이 10배가 됐다.
    # 2 epoch 은 underfit 이라 가장 강한 신호(투수 실력)만 배우는데 그건 CatBoost 도
    # 배우는 것이다. 더 학습해야 신경망 고유의 것을 배우고 그게 다양성이 된다.
    # 지인은 단독 성능 기준으로 2 를 골랐지만 앙상블 재료의 기준은 다르다.
    "ep8":       dict(ep1=8),
    "ep16":      dict(ep1=16),
    "ep4_sw2":   dict(ep1=4, decay=2.0),
    "ep8_sw2":   dict(ep1=8, decay=2.0),
    # ep4(+18.1) 와 ep8(+16.2) 사이에 봉우리가 있다. 그 자리를 좁힌다
    "ep3":       dict(ep1=3),
    "ep5":       dict(ep1=5),
    "ep6":       dict(ep1=6),
    # Stage2 를 더 세게 걸면 최근 시즌 적응이 늘어난다. 상관에 어떤 영향인지
    "ep4_s2x2":  dict(ep1=4, ep2=2),
    "ep4_s2lr":  dict(ep1=4, lr2=5e-4),
    # k 를 키우면 내부 앙상블 멤버가 늘어난다
    "ep4_k64":   dict(ep1=4, k=64),
}


if __name__ == "__main__":
    want = sys.argv[1] if len(sys.argv) > 1 else "base"
    # 인자에 :s3 를 붙이면 시드 3개 평균 (예: ep8:s3)
    # TabM 은 k=32 가중치 공유라 MLP 시드앙상블을 망친 '붕괴 시드' 위험이 낮다.
    # 순수 분산 감소라 과적합 위험도 없다 (CatBoost 는 10시드로 +9.7 이었다).
    todo = list(CASES) if want == "all" else [want]
    res = []
    for nm in todo:
        if nm.endswith(":s3"):
            key = nm[:-3]
            ps = []
            for sd in (42, 1, 777):
                tr_ = run(f"{key}_s{sd}", seed=sd, **CASES[key])
                ps.append(np.load(os.path.join(OUT, f"tabm_{key}_s{sd}.npy")))
            avg = np.mean(ps, 0)
            np.save(os.path.join(OUT, f"tabm_{key}_s3.npy"), avg)
            solo = F.best_shift(avg[mm], yv[mm])[0]
            cc = np.corrcoef(avg[mm], CB[mm])[0, 1]
            bw, bs_ = 0.0, BASE
            for wv in np.arange(0.05, 0.85, 0.05):
                s_ = F.best_shift(((1 - wv) * CUR + wv * avg)[mm], yv[mm])[0]
                if s_ > bs_:
                    bs_, bw = s_, wv
            log(f"\n  [{key} 시드3평균]  단독 {solo:.1f}  CB상관 {cc:.4f}  "
                f"비중 {bw:.2f} -> {bs_:.1f} ({bs_-BASE:+.1f})  LB 예상 "
                f"{1021 + 0.97*(bs_-BASE):.0f}\n")
            res.append(dict(name=f"{key}:s3", solo=solo, cor_cb=cc, cor_mlp=0.0,
                            w=bw, blended=bs_, lb=1021 + 0.97 * (bs_ - BASE)))
        else:
            res.append(run(nm, **CASES[nm]))
        with open(os.path.join(OUT, "results.json"), "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    log("=" * 72)
    log(f"  {'구성':<12s}{'단독':>9s}{'CB상관':>9s}{'비중':>7s}{'섞은뒤':>9s}{'LB예상':>9s}")
    for r in res:
        log(f"  {r['name']:<12s}{r['solo']:9.1f}{r['cor_cb']:9.4f}"
            f"{r['w']:7.2f}{r['blended']:9.1f}{r['lb']:9.0f}")
