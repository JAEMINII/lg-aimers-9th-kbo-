# -*- coding: utf-8 -*-
"""MLP 시드를 늘리면 앙상블이 더 오르는가 — 관문(2023 학습 -> 2024)에서 측정.

왜 재보나
  corr_fc.py 기록: FC 신경망 시드 3개의 단독 점수가 506.8 / 545.8 / 712.1 인데
  3시드 평균은 785.5 였다. 단일 평균(588.2)보다 +197. 신경망은 시드 편차가 크고
  평균내면 그만큼 오른다. 지금 제출본의 MLP 는 시드 하나뿐이다.

  MLP 도 같다면 MLP 팔 자체가 좋아지고, 비중도 더 실을 수 있다.
  CatBoost 는 이미 시드 3개 평균이라 이 이득을 이미 먹었다.

측정
  2023 한 시즌만 학습 (제출본과 같은 구조, 한 시즌 앞당긴 관문)
  시드 1 은 season_2023만.npy 로 이미 있다. 2, 3, 4 를 추가한다.
"""
import os, sys, time
import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(SC, "mlp_seeds_progress.txt")


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
import gc
gc.collect()

i_fit = np.where(season == 2023)[0]
i_gt = np.where(season == 2024)[0]
mm = ~is_f[i_gt]
yv = y[i_gt].astype(np.float64)


def bss(p):
    r = yv[mm].mean()
    return 100000 * (1 - ((p[mm] - yv[mm]) ** 2).mean() / (r * (1 - r)))


def sh(p, c=0.0):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + c)))


log(f"학습 2023 {len(i_fit):,}행  -> 관문 2024 {len(i_gt):,}행 (1군 {int(mm.sum()):,})")

from pytabkit import MLP_PLR_D_Classifier as M     # noqa: E402

SEEDS = [2, 3, 4]
for sd in SEEDS:
    out = os.path.join(SC, f"season_2023만_s{sd}.npy")
    if os.path.exists(out):
        log(f"  시드 {sd} 이미 있음 — 건너뜀")
        continue
    t0 = time.time()
    m = M(random_state=sd, n_threads=4, verbosity=0, device="cpu", batch_size=512)
    m.fit(X[i_fit], y[i_fit].astype(int), cat_col_names=cat_idx)
    p = m.predict_proba(X[i_gt])[:, 1]
    np.save(out, p)
    del m
    gc.collect()
    log(f"  시드 {sd}  단독 shift0 {bss(sh(p)):7.1f}   예측평균 {p[mm].mean():.4f}   "
        f"({time.time()-t0:.0f}s)")

# ------------------------------------------------------------------ 집계
import json                                        # noqa: E402
A = {k: {kk: np.asarray(vv, np.float64) for kk, vv in v.items()}
     for k, v in json.load(open(os.path.join(SC, "blend_state.json"))).items()}
cb = 0.6 * A["CatBoost"]["full"] + 0.4 * A["CatBoost"]["r"]

ps = [np.load(os.path.join(SC, "season_2023만.npy")).astype(np.float64)]
for sd in SEEDS:
    f = os.path.join(SC, f"season_2023만_s{sd}.npy")
    if os.path.exists(f):
        ps.append(np.load(f).astype(np.float64))

log("\n" + "=" * 74)
log("시드별 단독 / 누적평균")
log("=" * 74)
for k in range(1, len(ps) + 1):
    avg = np.mean(ps[:k], 0)
    log(f"  시드 {k}개 평균   MLP 단독 {bss(sh(avg)):7.1f}   "
        f"CB 와 상관 {np.corrcoef(avg[mm], cb[mm])[0,1]:.4f}")

log("\n" + "=" * 74)
log("앙상블 (shift 0, 관문 1군).  현 제출본 = 시드1개 x 0.20 = 리더보드 996")
log("=" * 74)
base = bss(sh(0.8 * cb + 0.2 * ps[0]))
log(f"  기준 관문 {base:.1f}   (관문 델타 x 0.36 = 리더보드 예상 델타)")
log(f"  {'시드수':>5s} {'w=0.20':>9s} {'0.25':>9s} {'0.30':>9s} {'0.35':>9s} {'0.40':>9s}"
    f" {'0.50':>9s}   최적w")
for k in range(1, len(ps) + 1):
    avg = np.mean(ps[:k], 0)
    row, best, bw = [], -1e9, 0
    for w in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        s = bss(sh((1 - w) * cb + w * avg))
        row.append(f"{s:9.1f}")
        if s > best:
            best, bw = s, w
    log(f"  {k:5d} " + " ".join(row) + f"   {bw:.2f} -> 예상 {996+0.36*(best-base):.1f}")
