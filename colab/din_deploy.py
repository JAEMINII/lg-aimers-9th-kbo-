# -*- coding: utf-8 -*-
"""배치용 DIN-lite 를 학습하고 제출 패키지가 읽을 npz 로 내보낸다.

관문 (colab/din_gate.py)
    VS=2022  w=0.20 +20.8  3/3        VS=2024  w=0.20 +11.7  3/3
    두 폴드 같은 부호, 모든 가중 3/3. 판정선 통과.

MNCA 전례를 반드시 확인할 것
    MNCA 관문 단독 897.5 / 상관 0.954 / +8.2  ->  배치 1069->1045.
    서명은 관문상관 0.954 vs 배치상관 0.696 이었다.
    DIN 관문 상관은 0.9421 이다. 배치본으로 다시 재서 크게 벌어지면 내지 말 것.

내보내는 것
    din_{branch}_s{seed}.npz   가중치 (all/regular/futures x 3시드)
    din_meta.npz               전처리 통계 + 2025용 시퀀스 표
        전처리는 ctr_zoo.build_inputs 가 만든 것을 그대로 담는다.
        학습과 추론이 같은 표를 쓰게 하려는 것이다 — 이 프로젝트에서
        점수를 제일 많이 깎은 결함이 학습/추론 규약 불일치였다.
"""
import os
import sys
import time

import numpy as np
import torch

SC = os.path.dirname(os.path.abspath(__file__))
for p in (SC, "/root/aimers", os.path.dirname(SC)):
    if p not in sys.path:
        sys.path.insert(0, p)
OUT = os.environ.get("DIN_OUT", "/root/aimers/din_out")
os.makedirs(OUT, exist_ok=True)
SEEDS = tuple(int(x) for x in os.environ.get("DD_SEEDS", "42,1,777").split(","))

import ctr_zoo as Z                                             # noqa: E402

if __name__ == "__main__":
    Z.VS = 2025                       # 학습 구간 = 2019~2024 전부
    D = Z.build_inputs()
    season, isf = D["season"], D["isf"]
    tr_idx = np.where(season < 2025)[0]
    old = season <= 2022
    w = np.where(isf & old, 0.1, 1.0)
    print(f"  학습 {len(tr_idx):,}행 (전 시즌)  시드 {SEEDS}", flush=True)
    dummy = np.arange(1, dtype=np.int64)
    for br, sel in (("all", np.ones(len(tr_idx), bool)),
                    ("regular", ~isf[tr_idx]), ("futures", isf[tr_idx])):
        for sd in SEEDS:
            t0 = time.time()
            m = Z.fit("DIN", D, tr_idx[sel], w, sd, dummy, return_model=True)
            sdict = {k.replace(".", "__"): v.detach().cpu().numpy()
                     for k, v in m.state_dict().items()}
            np.savez_compressed(os.path.join(OUT, f"din_{br}_s{sd}.npz"), **sdict)
            n = sum(v.size for v in sdict.values())
            print(f"    {br:8s} seed {sd:>3d}  {time.time()-t0:5.0f}s  "
                  f"파라미터 {n:,}", flush=True)
            del m
            torch.cuda.empty_cache()

    meta = dict(cards=np.asarray(D["cards"], np.int64),
                n_state=np.int64(D["n_state"]), n_cell=np.int64(D["n_cell"]),
                n_num=np.int64(D["Xn"].shape[1]), kseq=np.int64(Z.KSEQ),
                emb=np.int64(Z.EMB),
                cols=D["P_cols"], ci=D["P_ci"], ni=D["P_ni"],
                med=D["P_med"], mu=D["P_mu"], sd=D["P_sd"], hn=D["P_hn"],
                dep_state=D["DEP_S"], dep_cell=D["DEP_C"], dep_mask=D["DEP_M"],
                pids=D["PIDS"])
    for a, u in enumerate(D["P_uvals"]):
        meta[f"uval_{a}"] = np.asarray(u, np.float64)
    np.savez_compressed(os.path.join(OUT, "din_meta.npz"), **meta)
    sz = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print(f"  meta 저장.  총 {sz/1024/1024:.2f} MB  ({len(os.listdir(OUT))}개 파일)")
    print(f"  시퀀스 표 (2025용)  투수 {len(D['PIDS'])}명  "
          f"평균길이 {D['DEP_M'].sum(1).mean():.1f}/{Z.KSEQ}")
