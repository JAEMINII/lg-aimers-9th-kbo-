# -*- coding: utf-8 -*-
"""TabM 재시도 — 이번엔 pytabkit 구현으로, 최근 시즌만 학습해서.

v1/v2 (직접 짠 torch) 가 막힌 지점
  v1  2019~2023 전체 학습. 관문 742.6 이 최고. CatBoost 850 에 한참 못 미침
  v2  '내부검증과 관문이 정반대로 움직인다' 에서 막힘.
      내부검증 Brier 최저 에포크가 관문 최악이었다.

그 뒤에 알게 된 것 두 가지가 v1/v2 에는 안 들어가 있다
  1  신경망은 최근 한 시즌만 학습해야 한다 (MLP 455.3 -> 771.9)
     전체를 주면 옛 리그를 외운다. TabM 도 드리프트 장치가 없으니 같을 것이다
  2  pytabkit 기본값은 체크포인트/조기종료를 정확도로 건다
     확률 채점 문제에서 이건 잡음이다. val_metric_name='cross_entropy' 로 바꾼다
     (v2 가 본 '내부검증이 관문과 반대' 도 같은 종류의 문제였을 수 있다)

pytabkit 판을 쓰는 이유
  TabM_D_Classifier 의 기본 전처리가 tfms=['quantile_tabr'] 로 MLP-PLR 과 같다.
  즉 이번에 만든 numpy 추출 경로(분위수변환 / PLR / 범주임베딩)가 그대로 재사용된다.
  새로 짜야 하는 건 BatchEnsemble 의 rank-1 어댑터뿐인데 원소곱이라 쉽다.

  num_emb_type='plr' 을 주면 TabReD 가 최고라 한 조합(TabM + PLR)이 된다.
  기본값 'none' 과 둘 다 잰다.

넘어야 할 선 (관문 2024, 1군, shift 0)
  MLP-PLR 2023만      773.3
  CatBoost 60+40      850.6
  현 제출본 앙상블      876.3  = 리더보드 996
목표는 TabM 단독 점수가 아니라 **CatBoost 와의 상관**이다.
MLP 가 상관 0.889 로 +9 를 만들었다. 그보다 낮으면 더 먹을 수 있다.
"""
import os, sys, time, gc, json
import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(SC, "tabm_pytk_progress.txt")


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

A = {k: {kk: np.asarray(vv, np.float64) for kk, vv in v.items()}
     for k, v in json.load(open(os.path.join(SC, "blend_state.json"))).items()}
cb = 0.6 * A["CatBoost"]["full"] + 0.4 * A["CatBoost"]["r"]
mlp = np.load(os.path.join(SC, "season_2023만.npy")).astype(np.float64)


def bss(p):
    r = yv[mm].mean()
    return 100000 * (1 - ((p[mm] - yv[mm]) ** 2).mean() / (r * (1 - r)))


def sh(p, c=0.0):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + c)))


BASE = bss(sh(0.8 * cb + 0.2 * mlp))
log(f"학습 2023 {len(i_fit):,}행 -> 관문 2024 (1군 {int(mm.sum()):,})")
log(f"기준: 현 제출본 관문 {BASE:.1f} = 리더보드 996  (관문델타 x0.36 = LB델타)")

from pytabkit import TabM_D_Classifier as T          # noqa: E402

CASES = [
    ("TabM 기본",     dict(num_emb_type="none")),
    ("TabM+PLR",      dict(num_emb_type="plr")),
]
for name, kw in CASES:
    out = os.path.join(SC, f"tabm_{name.replace(' ', '_').replace('+', '_')}.npy")
    if os.path.exists(out):
        log(f"  {name} 이미 있음 — 건너뜀")
        continue
    t0 = time.time()
    try:
        m = T(random_state=1, n_threads=3, verbosity=0, device="cpu",
              val_metric_name="cross_entropy", **kw)
        m.fit(X[i_fit], y[i_fit].astype(int), cat_col_names=cat_idx)
        p = m.predict_proba(X[i_gt])[:, 1]
    except Exception as e:
        log(f"  {name:12s} 실패: {type(e).__name__} {str(e)[:160]}")
        continue
    np.save(out, p)
    del m
    gc.collect()
    log(f"  {name:12s} 단독 {bss(sh(p)):8.1f}   CB상관 "
        f"{np.corrcoef(p[mm], cb[mm])[0,1]:.4f}   MLP상관 "
        f"{np.corrcoef(p[mm], mlp[mm])[0,1]:.4f}   예측평균 {p[mm].mean():.4f}   "
        f"({time.time()-t0:.0f}s)")

# ------------------------------------------------------------------ 집계
log("\n" + "=" * 80)
log("앙상블 (shift 0)")
log("=" * 80)


def show(tag, p):
    d = bss(sh(p)) - BASE
    log(f"  {tag:44s} {bss(sh(p)):7.1f}  ({d:+6.1f})  예상 {996+0.36*d:6.1f}")


for name, _ in CASES:
    f = os.path.join(SC, f"tabm_{name.replace(' ', '_').replace('+', '_')}.npy")
    if not os.path.exists(f):
        continue
    tb = np.load(f).astype(np.float64)
    log(f"\n[{name}]")
    for w in (0.15, 0.20, 0.25, 0.30, 0.40):
        show(f"{1-w:.2f} CB + {w:.2f} TabM  (MLP 없이)", (1 - w) * cb + w * tb)
    for wt in (0.10, 0.15, 0.20):
        show(f"{0.8-wt:.2f} CB + 0.20 MLP + {wt:.2f} TabM", (0.8 - wt) * cb + 0.2 * mlp + wt * tb)
    for wt in (0.15, 0.20):
        show(f"{0.7-wt:.2f} CB + 0.30 MLP + {wt:.2f} TabM", (0.7 - wt) * cb + 0.3 * mlp + wt * tb)
