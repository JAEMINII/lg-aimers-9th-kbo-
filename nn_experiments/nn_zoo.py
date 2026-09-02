# -*- coding: utf-8 -*-
"""신경망을 하나 더 찾는다 — 다양성의 원천은 모델 구조였다.

오늘 확인된 상관 구조
    모델 다름 + 데이터 다름   CatBoost <-> MLP(2023만)      0.9072
    모델 다름 + 데이터 같음   CB(2023만) <-> MLP(2023만)    0.9155
    모델 같음 + 데이터 다름   CB(시즌가중) <-> CB(균등)      0.9876
                            CB(시즌가중) <-> CB(2023만)    0.9686
    트리 계열끼리            LGB / HistGB / ExtraTrees      0.952~0.963

  같은 CatBoost 에 데이터만 바꾸면 상관이 0.97~0.99 로 거의 안 떨어진다(앙상블 +0.5).
  트리 안에서는 배깅·선형까지 내려가도 0.95 아래로 안 간다(앙상블 +1.1 이하).
  0.90 대를 만든 건 신경망뿐인데 우리는 신경망을 하나만 쓰고 있다.

  => MLP 의 다양성은 모델 구조에서 나왔다. 시즌을 자른 건 MLP 를 쓸 만하게
     만든 것(455 -> 772)이지 다양성의 원천이 아니다.

그래서 다른 신경망 구조를 잰다
    RealMLP_TD      pytabkit 대표 모델. MLP 계열이지만 전처리/스케줄이 많이 다르다
    Resnet_RTDL     잔차 연결. MLP 와 구조가 다르다
    TabM_D (k=8)    BatchEnsemble. k=32 는 CPU 에서 3h43m 무산출이라 8 로 낮춘다
    RealTabR_D      검색 기반(kNN 유사). 트리/MLP 와 근본적으로 다르다

전부 2023 한 시즌만 학습한다 (MLP 에서 확인된 조건).
보는 것은 단독 점수가 아니라 **CatBoost 및 기존 MLP 와의 상관**이다.
"""
import os, sys, time, gc, traceback
import numpy as np
from scipy.optimize import minimize_scalar

SC = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(SC, "nn_zoo_progress.txt")


def log(s):
    print(s, flush=True)
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(s + "\n")


z = np.load(os.path.join(SC, "nn_cache.npz"), allow_pickle=False)
Xc, Xn, y, season, is_f = z["Xc"], z["Xn"], z["y"], z["season"], z["is_f"]
X = np.concatenate([Xc.astype(np.float32), Xn], 1)
cat_idx = list(range(Xc.shape[1]))
del Xc, Xn
z.close()
gc.collect()

i_fit = np.where(season == 2023)[0]
i_gt = np.where(season == 2024)[0]
mm = ~is_f[i_gt]
yv = y[i_gt].astype(np.float64)


def bss(p):
    r = yv[mm].mean()
    return 100000 * (1 - ((p[mm] - yv[mm]) ** 2).mean() / (r * (1 - r)))


def best(p):
    def sh(q, c):
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return 1 / (1 + np.exp(-(np.log(q / (1 - q)) + c)))
    r = minimize_scalar(lambda c: -bss(sh(p, c)), bounds=(-0.3, 0.3), method="bounded")
    return -r.fun


CB = np.load(os.path.join(SC, "alpha_50.npy"))                      # 시즌가중 CatBoost
MLP = np.load(os.path.join(SC, "season_2023만.npy")).astype(np.float64)
CUR = 0.8 * CB + 0.2 * MLP
cur_s = best(CUR)
log(f"학습 2023 {len(i_fit):,}행 -> 관문 2024 1군 {int(mm.sum()):,}")
log(f"기준: 현 제출본 {cur_s:.1f}   (CatBoost 단독 {best(CB):.1f} / MLP 단독 {best(MLP):.1f})")
log(f"참고: CatBoost~MLP 상관 {np.corrcoef(CB[mm], MLP[mm])[0,1]:.4f}\n")
log(f"  {'모델':<16s}{'단독':>9s}{'CB상관':>9s}{'MLP상관':>9s}{'최적비중':>9s}"
    f"{'얹은결과':>10s}{'대비':>8s}{'시간':>8s}")

import pytabkit as P                                                 # noqa: E402

CASES = [
    ("RealMLP_TD",  lambda: P.RealMLP_TD_Classifier(
        random_state=1, n_threads=6, verbosity=0, device="cpu")),
    ("Resnet_RTDL", lambda: P.Resnet_RTDL_D_Classifier(
        random_state=1, n_threads=6, verbosity=0, device="cpu", batch_size=512,
        val_metric_name="cross_entropy")),
    ("TabM_k8",     lambda: P.TabM_D_Classifier(
        random_state=1, n_threads=6, verbosity=0, device="cpu",
        tabm_k=8, n_epochs=40, patience=8)),
    ("RealTabR",    lambda: P.RealTabR_D_Classifier(
        random_state=1, n_threads=6, verbosity=0, device="cpu")),
]

for name, mk in CASES:
    out = os.path.join(SC, f"nnzoo_{name}.npy")
    if os.path.exists(out):
        p = np.load(out)
    else:
        t0 = time.time()
        try:
            m = mk()
            m.fit(X[i_fit], y[i_fit].astype(int), cat_col_names=cat_idx)
            p = m.predict_proba(X[i_gt])[:, 1]
            del m; gc.collect()
        except Exception as e:
            log(f"  {name:<16s} 실패: {type(e).__name__} {str(e)[:100]}")
            with open(os.path.join(SC, "nn_zoo_errors.txt"), "a", encoding="utf-8") as f:
                f.write(f"\n=== {name}\n{traceback.format_exc()}")
            continue
        np.save(out, p)
        el = time.time() - t0
    p = p.astype(np.float64)
    el = locals().get("el", 0.0)
    bw, bs = 0.0, cur_s
    for wv in np.arange(0.05, 0.55, 0.05):
        s = best((1 - wv) * CUR + wv * p)
        if s > bs:
            bs, bw = s, wv
    log(f"  {name:<16s}{best(p):9.1f}{np.corrcoef(p[mm], CB[mm])[0,1]:9.4f}"
        f"{np.corrcoef(p[mm], MLP[mm])[0,1]:9.4f}{bw:9.2f}{bs:10.1f}{bs-cur_s:+8.1f}{el:7.0f}s")
    locals().pop("el", None)

log("\n판독: CB상관 0.93 아래면 MLP 급 재료다. '대비' +8 이상이면 넣을 값어치가 있다.")
log("      MLP 상관도 같이 본다 — 기존 MLP 와 겹치면 둘을 같이 넣어도 소용없다.")
