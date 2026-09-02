# -*- coding: utf-8 -*-
"""TabM 수치임베딩을 바꿔서 비교한다.

왜
    지금 우리 TabM 은 num_embeddings='linear_relu' 다. 지인 selected_config 를
    그대로 받은 것이지 우리가 비교해서 고른 게 아니다. TabM 논문은 주기임베딩과
    구간선형임베딩도 함께 제시하고, 데이터에 따라 그쪽이 낫다고 보고한다.
    우리 flatMLP 가 PLR(주기+선형+ReLU)로 잘 돌아가는 것도 근거가 된다.

    그리고 이 비교는 오늘 확인한 관문의 약점을 피해 간다. 관문은 계열이 다르면
    순위를 틀리고(CatBoost 903 > TabM 868 인데 리더보드는 반대) epoch 도 틀렸다.
    임베딩만 바꾸는 건 같은 TabM 계열 안의 변형이라 그 함정이 없다.

세 벌
    linear_relu   현 설정
    periodic      PeriodicEmbeddings — 주기 특징 후 선형. 우리 flatMLP 방식
    piecewise     PiecewiseLinearEmbeddings — 분위수 구간. 논문에서 자주 최고
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


def make(kind, n_num, cards, d_emb=16, k=32):
    import rtdl_num_embeddings as rne
    from tabm import TabM
    if kind == "linear_relu":
        emb = rne.LinearReLUEmbeddings(n_num, d_embedding=d_emb)
    elif kind == "periodic":
        emb = rne.PeriodicEmbeddings(n_num, d_embedding=d_emb, lite=False)
    elif kind == "piecewise":
        # 분위수 구간은 학습 구간에서만 만든다
        x = torch.from_numpy(G.Xn[G.m_tr])
        bins = rne.compute_bins(x, n_bins=48)
        emb = rne.PiecewiseLinearEmbeddings(bins, d_embedding=d_emb,
                                            activation=False, version="B")
    else:
        raise ValueError(kind)
    return TabM.make(n_num_features=n_num,
                     cat_cardinalities=[int(c) for c in cards], d_out=1,
                     num_embeddings=emb, arch_type="tabm", k=k,
                     n_blocks=3, d_block=256, dropout=0.1).to(G.DEV)


def one(kind, seed, cards):
    P = {}
    tr_idx = np.where(G.m_tr)[0]
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("futures", G.is_f[tr_idx]), ("regular", ~G.is_f[tr_idx])):
        idx = tr_idx[sel]
        torch.manual_seed(seed)                 # 초기화까지 덮는다
        torch.cuda.manual_seed_all(seed)
        m = make(kind, G.Xn.shape[1], cards)
        G.train(m, idx, 2, 2e-3, seed=seed, tag=f"{kind} {br} S1")
        s2 = G.season[idx] == VS - 1
        G.train(m, idx[s2], 1, 2e-4, params=G.stage2_params(m), seed=seed,
                tag=f"{kind} {br} S2")
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

    G.log("\n" + "=" * 72)
    G.log("  임베딩          시드평균 단독   제출비중   개별 시드")
    G.log("=" * 72)
    ref = None
    for kind in ("linear_relu", "periodic", "piecewise"):
        try:
            t0 = time.time()
            ps = [one(kind, sd, cards) for sd in SEEDS]
        except Exception as e:
            G.log(f"  {kind:13s} 실패: {type(e).__name__} {str(e)[:80]}")
            continue
        solos = [F.best_shift(p[FULL], yv[FULL])[0] for p in ps]
        r = np.mean(ps, 0)
        solo = F.best_shift(r[FULL], yv[FULL])[0]
        bl = F.best_shift((0.10 * CB + 0.30 * MLPF + 0.60 * r)[FULL], yv[FULL])[0]
        np.save(os.path.join(OUT, f"emb_{kind}.npy"), r)
        dl = "" if ref is None else f"   기준대비 {solo-ref[0]:+6.1f} / {bl-ref[1]:+6.1f}"
        if ref is None:
            ref = (solo, bl)
        G.log(f"  {kind:13s} {solo:7.1f}      {bl:7.1f}   "
              f"[{', '.join(f'{v:.0f}' for v in solos)}]   {time.time()-t0:.0f}s{dl}")
    G.log("\n  +5 미만은 판정 보류. 같은 계열 안의 비교라 관문을 믿을 수 있다.")
