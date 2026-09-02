# -*- coding: utf-8 -*-
"""ep2 위에서 TabM 시드 앙상블이 효과가 있는지 잰다.

왜 다시 하나
    시드 앙상블은 예전에 ep4/ep5 위에서만 재봤다. 그 뒤 ep4 가 리더보드에서
    980 으로 무너지고 ep2 가 맞는 베이스인 게 드러났는데, 시드는 다시 안 돌렸다.
    ep2 + 다중시드 조합은 한 번도 측정된 적이 없다.

채점은 전체(R+F) 기준이다
    tabm_gate_gpu.py 는 1군만 채점한다(mm = ~is_f). 그 기준이 ep4 를 좋아 보이게
    만들었고 실제로는 퓨처스가 무너져 980 이 나왔다. 여기서는 두 기준을 다 찍되
    판단은 전체로 한다.

설정은 submit_jaemin_14(리더보드 1041)와 같다
    Stage1 2epoch -> Stage2 마지막 시즌 1epoch, k=32, 시즌가중 없음
    all/futures/regular 3브랜치를 0.6:0.4 로 합침
"""
import os
import sys
import time

import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
OUT = "/workspace/aimers/out"
SEEDS = (42, 1, 777)

import features44 as F                                          # noqa: E402
import tabm_gate_gpu as G                                       # noqa: E402


def score(p, msk):
    """최적 시프트에서 잰다. 고정 시프트면 수준 보정과 판별력이 섞인다."""
    return F.best_shift(p[msk], G.yv[msk])[0]


def load_or_train(seed):
    """세 시드를 모두 같은 코드로 학습한다.

    out/ 에 br_base_*.npy (시드 42) 가 남아 있지만 재사용하지 않는다. 저장 이름
    규칙이 그 사이에 바뀌었고(br_ -> br2024_), 같은 코드에서 나온 건지 확인할
    방법이 없다. 시드당 188초라 다시 학습하는 편이 싸고 확실하다.
    """
    tag = f"base_s{seed}"
    paths = {br: os.path.join(OUT, f"br{G.VS}_{tag}_{br}.npy")
             for br in ("all", "futures", "regular")}
    if all(os.path.exists(v) for v in paths.values()):
        G.log(f"  seed {seed}: 이번 실행에서 만든 파일 재사용")
    else:
        t = time.time()
        G.run(tag, seed=seed)      # ep1=2, ep2=1, stage2=True, k=32 가 기본값
        G.log(f"  seed {seed}: 학습 {time.time()-t:.0f}s")
    return {br: np.load(v) for br, v in paths.items()}


if __name__ == "__main__":
    gate = G.gate
    is_f = G.is_f[gate]
    FULL = np.ones(len(gate), bool)          # 전체 채점 (R+F) — 이쪽이 리더보드와 맞는다
    ONE = ~is_f                              # 1군 채점 — 참고용

    CB = np.load(os.path.join(SC, "cb_gate.npy"))
    MLPF = np.load(os.path.join(OUT, "mlpnew_flat_s3.npy"))    # flatMLP 시드3 평균

    P = [load_or_train(s) for s in SEEDS]

    G.log("\n" + "=" * 68)
    G.log("시드 개수별   (전체채점 / 1군채점)")
    G.log("=" * 68)
    rows = []
    for n in (1, 2, 3):
        avg = {br: np.mean([p[br] for p in P[:n]], 0)
               for br in ("all", "futures", "regular")}
        routed = np.where(is_f,
                          0.4 * avg["futures"] + 0.6 * avg["all"],
                          0.4 * avg["regular"] + 0.6 * avg["all"])
        solo_f, solo_1 = score(routed, FULL), score(routed, ONE)
        # 제출본과 같은 비중으로 섞었을 때
        blend = 0.10 * CB + 0.30 * MLPF + 0.60 * routed
        bl_f = score(blend, FULL)
        # 비중을 다시 훑는다. 시드가 늘면 최적 비중도 움직인다
        best = (-1e9, None)
        for w in np.arange(0.30, 0.91, 0.05):
            rest = 1.0 - w
            q = (rest / 4.0) * CB + (rest * 3.0 / 4.0) * MLPF + w * routed
            s_ = score(q, FULL)
            if s_ > best[0]:
                best = (s_, w)
        G.log(f"  시드 {n}   단독 {solo_f:7.1f} / {solo_1:7.1f}    "
              f"제출비중 {bl_f:7.1f}    최적 TabM {best[1]:.2f} -> {best[0]:7.1f}")
        rows.append((n, solo_f, bl_f, best[0], best[1]))

    G.log("\n  시드 1 -> 3 변화")
    G.log(f"    단독      {rows[2][1] - rows[0][1]:+6.1f}")
    G.log(f"    제출비중  {rows[2][2] - rows[0][2]:+6.1f}")
    G.log(f"    최적비중  {rows[2][3] - rows[0][3]:+6.1f}")
    G.log("\n  주의: 관문→리더보드 환산은 계열을 섞으면 빗나간다(1076 추정 -> 실제 1046).")
    G.log("  여기서는 '시드를 늘리면 오르는가' 만 본다. 절대값 예측은 하지 않는다.")
