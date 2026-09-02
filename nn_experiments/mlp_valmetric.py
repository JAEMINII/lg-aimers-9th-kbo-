# -*- coding: utf-8 -*-
"""체크포인트 기준을 정확도 -> 교차엔트로피로 바꾼다. 관문(2023 학습 -> 2024).

왜
  mlp_seeds.py 로 시드 3개를 더 돌렸더니 단독 점수가
      시드1  773.3     시드2 -2759.3     시드3  567.8     시드4  799.8
  로 흩어졌다. 평균내면 오히려 나빠진다(시드1개 773 -> 4개 평균 570).

  학습 로그를 보면 원인이 분명하다. pytabkit 의 기본값 val_metric_name='class_error'
  는 체크포인트와 조기종료를 **정확도(valid_acc)** 로 건다.
      시드2   valid_loss 최저는 4에폭(0.6873). 그런데 valid_acc 최고가 18에폭이라
              18에폭을 저장했다. 그때 valid_loss 는 0.6949 로 이미 과적합 구간이다
      시드3   5에폭 저장 -> 멀쩡
      시드4   3에폭 저장 -> 멀쩡
  채점이 브라이어(확률 제곱오차)인데 모델 선택을 정확도로 하고 있었다.
  지금 제출본의 시드1이 좋은 건 운 좋게 일찍 걸린 것이다.

  val_metric_name='cross_entropy' 로 주면 둘 다 valid_loss 기준이 된다
  (pytabkit/models/nn_models/rtdl_resnet.py:1045-1072 에서 확인).

기대
  시드마다 같은 지점에서 멈추므로 편차가 줄고, 그러면 시드 평균이 제대로 듣는다.
"""
import os, sys, time, gc, json
import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(SC, "mlp_valmetric_progress.txt")


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


def sh(p, c=0.0):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + c)))


log(f"학습 2023 {len(i_fit):,}행 -> 관문 2024 (1군 {int(mm.sum()):,})  "
    f"val_metric_name='cross_entropy'")

from pytabkit import MLP_PLR_D_Classifier as M     # noqa: E402

SEEDS = [1, 2, 3]
for sd in SEEDS:
    out = os.path.join(SC, f"ce2023_s{sd}.npy")
    if os.path.exists(out):
        log(f"  시드 {sd} 이미 있음 — 건너뜀")
        continue
    t0 = time.time()
    m = M(random_state=sd, n_threads=4, verbosity=0, device="cpu", batch_size=512,
          val_metric_name="cross_entropy")
    m.fit(X[i_fit], y[i_fit].astype(int), cat_col_names=cat_idx)
    p = m.predict_proba(X[i_gt])[:, 1]
    np.save(out, p)
    del m
    gc.collect()
    log(f"  시드 {sd}  단독 {bss(sh(p)):8.1f}   예측평균 {p[mm].mean():.4f}   "
        f"({time.time()-t0:.0f}s)")

# ------------------------------------------------------------------ 집계
A = {k: {kk: np.asarray(vv, np.float64) for kk, vv in v.items()}
     for k, v in json.load(open(os.path.join(SC, "blend_state.json"))).items()}
cb = 0.6 * A["CatBoost"]["full"] + 0.4 * A["CatBoost"]["r"]
old = np.load(os.path.join(SC, "season_2023만.npy")).astype(np.float64)   # 현 제출본의 MLP
base = bss(sh(0.8 * cb + 0.2 * old))

ps = []
for sd in SEEDS:
    f = os.path.join(SC, f"ce2023_s{sd}.npy")
    if os.path.exists(f):
        ps.append(np.load(f).astype(np.float64))

log("\n" + "=" * 78)
log("정확도 기준(기존) vs 교차엔트로피 기준(새로)")
log("=" * 78)
log(f"  기존 시드1 (현 제출본)   단독 {bss(sh(old)):8.1f}   CB 상관 "
    f"{np.corrcoef(old[mm], cb[mm])[0,1]:.4f}")
for k in range(1, len(ps) + 1):
    avg = np.mean(ps[:k], 0)
    log(f"  새 기준 시드 {k}개 평균   단독 {bss(sh(avg)):8.1f}   CB 상관 "
        f"{np.corrcoef(avg[mm], cb[mm])[0,1]:.4f}")

log("\n" + "=" * 78)
log(f"CatBoost 와 앙상블 (shift 0).  기준 = 현 제출본 관문 {base:.1f} = 리더보드 996")
log("=" * 78)
log(f"  {'구성':>16s} " + " ".join(f"{f'w={w:.2f}':>9s}" for w in
                                   (0.20, 0.25, 0.30, 0.35, 0.40, 0.50)) + "   최적 -> 예상")
def row(tag, mlp):
    vals, best, bw = [], -1e9, 0
    for w in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        s = bss(sh((1 - w) * cb + w * mlp))
        vals.append(f"{s:9.1f}")
        if s > best:
            best, bw = s, w
    log(f"  {tag:>16s} " + " ".join(vals) + f"   {bw:.2f} -> {996+0.36*(best-base):6.1f}")
row("기존 시드1", old)
for k in range(1, len(ps) + 1):
    row(f"새 {k}시드평균", np.mean(ps[:k], 0))
if ps:
    row("새3시드+기존", np.mean(ps + [old], 0))
