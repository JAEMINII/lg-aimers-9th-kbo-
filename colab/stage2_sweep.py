# -*- coding: utf-8 -*-
"""아직 안 재본 세 축을 한 번에 훑는다.

왜 이 셋인가
    Stage2 에폭   퓨처스 Stage2 는 2024 퓨처스 약 3만행 / 배치 2048 = 15배치뿐이다.
                  에폭 1이면 파라미터 갱신이 15번이다. 우리가 잰 'ep4' 는 Stage1 만
                  바꾼 것이라(ep1=4) 이 축은 미측정이다.
                  참고로 Stage1 을 4로 올리는 건 이미 태워서 실패했다(LB 1041 -> 980).
    저카디널리티 범주화
                  cat_idx 는 [4,5,12,15,16,17,18,19,20] 아홉 개뿐이고
                  inning(13종) balls(4) strikes(3) outs(3) month(8) dow(7) 은
                  수치형으로 들어간다. 1회와 2회의 차이가 선형이라는 가정이 깔린다.
    Stage1/Stage2 혼합
                  지금은 Stage2 체크포인트만 쓴다. Stage2 가 마지막 시즌에
                  과적합했는지 확인할 방법이 없다. 둘을 섞으면 그 비중이 답을 준다.

관문 신뢰도
    Stage2 에폭은 '학습량' 축이라 관문이 순위를 틀릴 수 있다(ep2/ep4 에서 그랬다).
    그래서 이 결과만으로 제출을 정하지 않는다. 나머지 둘은 학습량이 같은 변형이라
    관문을 믿을 수 있다.

시드 4개로 짝지어 본다. 비슷한 설정끼리 비교하면 짝지은 차이의 산포가 작다
(변조 실험에서 5.4였다). 유망한 것만 6~8시드로 확인한다.
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
DECAY = 3.5
VS = 2024

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402

gate, is_f, yv = G.gate, G.is_f[G.gate], G.yv
FULL = np.ones(len(gate), bool)


def sc(p):
    return F.best_shift(p[FULL], yv[FULL])[0]


def run(seed, ep1=2, ep2=1, k=32, decay=DECAY, keep_stage1=False):
    """3브랜치를 학습한다. keep_stage1 이면 Stage1 예측도 같이 돌려준다."""
    P1, P2 = {}, {}
    tr_idx = np.where(G.m_tr)[0]
    W = None if not decay else decay ** (G.season[tr_idx].astype(np.float64) - 2019)
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("futures", G.is_f[tr_idx]), ("regular", ~G.is_f[tr_idx])):
        idx = tr_idx[sel]
        w = None if W is None else W[sel]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        m = G.make_model(k=k)
        G.train(m, idx, ep1, 2e-3, w=w, seed=seed, tag=f"{br} S1")
        if keep_stage1:
            P1[br] = G.predict(m, gate)
        s2 = G.season[idx] == VS - 1
        G.train(m, idx[s2], ep2, 2e-4, params=G.stage2_params(m),
                w=None if w is None else w[s2], seed=seed, tag=f"{br} S2")
        P2[br] = G.predict(m, gate)
        del m
        torch.cuda.empty_cache()

    def route(P):
        return np.where(is_f, 0.4 * P["futures"] + 0.6 * P["all"],
                        0.4 * P["regular"] + 0.6 * P["all"])
    return (route(P2), route(P1) if keep_stage1 else None)


if __name__ == "__main__":
    d = F.build(DATA, VS=VS)
    F44 = list(d["F44"])
    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    MLPF = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))

    def blend(r):
        return sc(0.10 * CB + 0.30 * MLPF + 0.60 * r)

    def load_X(extra_cat):
        ci = list(d["cat_idx"])
        if extra_cat:
            ci = sorted(set(ci) | {F44.index(c) for c in extra_cat})
        Xn, Xc, cards = G.prep(d["X44"], G.m_tr, ci)
        G.Xn, G.cards = Xn, cards
        G.XN, G.XC = torch.from_numpy(Xn), torch.from_numpy(Xc)
        return len(ci)

    LOWCARD = ["game_month", "game_dayofweek", "inning",
               "balls_before", "strikes_before", "outs_before"]

    plans = [
        ("기준 s2=1", dict(), None),
        ("s2=2", dict(ep2=2), None),
        ("s2=3", dict(ep2=3), None),
        ("저카디널리티 범주화", dict(), LOWCARD),
    ]
    res = {}
    G.log("")
    G.log("  구성                  시드평균 단독   제출비중   짝차이평균±표준오차")
    for name, kw, extra in plans:
        ncat = load_X(extra)
        t0 = time.time()
        ps = [run(sd, **kw)[0] for sd in SEEDS]
        res[name] = ps
        r = np.mean(ps, 0)
        base = res.get("기준 s2=1")
        if base is not None and name != "기준 s2=1":
            dif = [sc(a) - sc(b) for a, b in zip(ps, base)]
            se = np.std(dif, ddof=1) / np.sqrt(len(dif))
            tail = f"   {np.mean(dif):+6.1f} ± {se:.1f}  양수 {sum(1 for x in dif if x>0)}/{len(dif)}"
        else:
            tail = f"   (범주 {ncat}개)"
        G.log(f"  {name:20s} {sc(r):9.1f}   {blend(r):9.1f}{tail}   "
              f"{time.time()-t0:.0f}s")

    # Stage1/Stage2 혼합은 예측을 따로 받아야 한다
    G.log("")
    G.log("  Stage1 / Stage2 혼합")
    load_X(None)
    pairs = [run(sd, keep_stage1=True) for sd in SEEDS]
    p2 = np.mean([a for a, _ in pairs], 0)
    p1 = np.mean([b for _, b in pairs], 0)
    G.log(f"    Stage1 단독 {sc(p1):7.1f}   Stage2 단독 {sc(p2):7.1f}")
    for w in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        q = w * p2 + (1 - w) * p1
        G.log(f"    Stage2 {w:.1f} / Stage1 {1-w:.1f}   단독 {sc(q):7.1f}   "
              f"제출비중 {blend(q):7.1f}")
    G.log("")
    G.log("  Stage2 에폭은 '학습량' 축이라 관문이 순위를 틀린 전례가 있다")
    G.log("  (ep2/ep4 에서 관문 +5.5, 리더보드 -61). 이 결과만으로 제출을 정하지 않는다.")
