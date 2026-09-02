# -*- coding: utf-8 -*-
"""TabPFN 속도와 점수를 먼저 잰다. 배치 가능 여부가 여기서 갈린다.

배치 예산 (규정 원문)
    추론 10분, 시험 245,789행, L4 GPU 22.4GiB.
    TabPFN 은 in-context 라 시험 행마다 문맥(=학습 부분표본)을 함께 통과시킨다.
    그래서 비용이 (문맥 크기) x (시험 행수) 로 붙는다. 이게 되는지가 먼저다.

    여기서는 3060 으로 재고 L4 로 환산한다. L4 가 약 1.5~2배 빠르므로
    3060 에서 10분을 넘기면 L4 에서도 위험하다.

무엇을 보나
    ① 문맥 1만 행으로 2만 행 예측에 몇 초 걸리나 -> 245,789행 환산
    ② 그 점수가 얼마나 되나 -> 학습곡선의 1만 행 지점과 비교할 값
    ③ 배깅(문맥을 여러 벌 뽑아 평균)이 점수를 얼마나 올리나

    피처는 features44 의 X44 를 쓴다. 탐색 단계라 지인 전처리까지 맞출 필요가
    없고, CPU 로만 올라와서 도는 관문과 GPU 를 덜 다툰다.
"""
import os
import sys
import time

import numpy as np
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, "/workspace/aimers")
DATA = "/workspace/aimers/data"
OUT = "/workspace/aimers/out"
VS = 2024
N_CTX = 10_000
N_EVAL = 20_000
TEST_ROWS = 245_789          # 규정에 적힌 실제 시험 행수

import features44 as F                                          # noqa: E402
from tabpfn import TabPFNClassifier                             # noqa: E402


def sc(p, y):
    from features44 import best_shift
    return best_shift(p, y)[0]


if __name__ == "__main__":
    d = F.build(DATA, VS=VS)
    X = d["X44"].astype(np.float32)
    y = d["y"].astype(np.int64)
    m_tr, m_va = d["m_tr"], d["m_va"]
    season = d["season"]
    ci = list(d["cat_idx"])
    F44 = list(d["F44"])
    med = np.nanmedian(X[m_tr], 0)
    X = np.where(np.isnan(X), med, X).astype(np.float32)

    tr_idx = np.where(m_tr)[0]
    va_idx = np.where(m_va)[0]
    rng = np.random.default_rng(0)
    ev = np.sort(rng.choice(va_idx, size=min(N_EVAL, len(va_idx)),
                            replace=False))
    yv = y[ev].astype(np.float64)
    print(f"  문맥 후보 {len(tr_idx):,}   평가 {len(ev):,}   "
          f"실제 성공률 {yv.mean():.4f}")
    print(f"  범주 지정 {len(ci)}개: {', '.join(F44[j] for j in ci)}")
    print(f"  GPU {torch.cuda.get_device_name(0)}\n")

    def draw(seed):
        """시즌 비율을 유지한 문맥 부분표본."""
        r = np.random.default_rng(seed)
        out = []
        s = season[tr_idx]
        for v in np.unique(s):
            pool = tr_idx[s == v]
            k = max(1, int(round(N_CTX * len(pool) / len(tr_idx))))
            out.append(r.choice(pool, size=min(k, len(pool)), replace=False))
        return np.concatenate(out)

    accs, times = [], []
    for b in range(4):
        c = draw(100 + b)
        t0 = time.time()
        import inspect as _I
        ok = set(_I.signature(TabPFNClassifier.__init__).parameters)
        kw = {"device": "cuda"}
        for k, v in (("n_estimators", 4), ("categorical_features_indices", ci),
                     ("ignore_pretraining_limits", True)):
            if k in ok:
                kw[k] = v
        clf = TabPFNClassifier(**kw)
        clf.fit(X[c], y[c])
        p = clf.predict_proba(X[ev])[:, 1].astype(np.float64)
        el = time.time() - t0
        times.append(el)
        accs.append(p)
        cur = np.mean(accs, 0)
        print(f"  배깅 {b+1}벌  문맥 {len(c):,}  {el:6.1f}s   "
              f"단독 {sc(p, yv):7.1f}   누적평균 {sc(cur, yv):7.1f}")
        del clf
        torch.cuda.empty_cache()

    per = float(np.mean(times))
    scale = TEST_ROWS / len(ev)
    print(f"\n  {len(ev):,}행 예측에 {per:.1f}s  ->  "
          f"{TEST_ROWS:,}행 환산 {per*scale/60:.1f}분 / 배깅 1벌")
    print(f"  배깅 4벌이면 {per*scale*4/60:.1f}분.  예산은 10분이고 "
          f"L4 는 3060 보다 약 1.5~2배 빠르다.")
    np.save(os.path.join(OUT, "tabpfn_smoke.npy"), np.mean(accs, 0))
    np.save(os.path.join(OUT, "tabpfn_smoke_idx.npy"), ev)
    print("\n  다음: 이 점수를 학습곡선의 1만 행 지점과 나란히 놓는다.")
    print("  TabPFN 이 그보다 크게 못 이기면 사전분포가 이 문제에 안 맞는 것이고,")
    print("  이기더라도 122만 행 모델(관문 906)과의 격차가 본론이다.")
