# -*- coding: utf-8 -*-
"""
공식 구현(pytabkit)으로 MLP-PLR / RealMLP 재시험.

왜 재구현을 버렸나
  내 손구현은 논문 구조의 근사치였고, 편차가 결과를 좌우할 만했다.
    PLR   sigma 미튜닝(0.1 로 찍음) · 분위수변환 대신 표준화 · d_emb 8
    TabM  k=8(논문 32) · 전층 어댑터(논문 주력은 TabM_mini) · BatchNorm 추가
          OneCycle 8에포크(논문은 상수 lr + patience 16)
  실제로 v2 곡선이 ep02 748 -> ep05 453 으로 무너졌는데, 이건 방법이 아니라
  내 설정을 시험한 것이다. CatBoost 를 −539 로 잘못 기각했던 것과 같은 자리다.

  pytabkit 은 논문 저자들이 여러 데이터셋에서 잡은 'tuned defaults(TD)' 를 준다.
  sigma 같은 민감한 값을 내가 찍을 필요가 없다.

측정 (지금까지와 동일 조건)
  관문 2019~2023 -> 2024, 1군 채점(2025 는 전부 1군)
  비교 대상 (1군, shift 0 / −0.05)
      FC 손구현 3시드     752.2 / 785.5
      sklearn 전체단독    814.2 / 856.2
      CatBoost 60+40      850.6 / 886.3

주의
  CPU 4스레드에 122만 행이라 느리다. 먼저 20만 행 부분표본으로 한 번 돌려
  시간과 대략적 수준을 재고, 쓸 만하면 전체로 올린다.
  (부분표본 점수는 전체보다 낮게 나오는 게 정상이다. 시간 측정이 목적이다)
"""
import os, sys, time
import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
def H(t):
    print("\n" + "=" * 86); print(t); print("=" * 86); sys.stdout.flush()

z = np.load(os.path.join(SC, "nn_cache.npz"), allow_pickle=False)
Xc, Xn, y, season, is_f = z["Xc"], z["Xn"], z["y"], z["season"], z["is_f"]
cat_names = z["cat_names"].tolist()
n_cat = Xc.shape[1]

# pytabkit 은 하나의 X 에서 범주형 열 인덱스를 지정받는다.
# 범주는 이미 0..card-1 정수로 인코딩돼 있고, 연속형은 표준화돼 있다.
# (PLR 은 내부에서 자체 전처리를 하므로 표준화 여부가 결정적이지 않다)
X = np.concatenate([Xc.astype(np.float32), Xn], 1)
cat_idx = list(range(n_cat))
print(f"X {X.shape}   범주 열 {n_cat}개 (앞쪽) + 연속 {Xn.shape[1]}개")

i_tr = np.where(season < 2024)[0]
i_gt = np.where(season == 2024)[0]
fv, yv = is_f[i_gt], y[i_gt]

def bss(p, yy):
    r = yy.mean()
    return 100000 * (1 - ((p - yy) ** 2).mean() / (r * (1 - r)))
def sh(p, c):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + c)))

SUB = int(os.environ.get("SUB_N", "200000"))     # 0 이면 전체
if SUB and SUB < len(i_tr):
    rng = np.random.RandomState(0)
    # 시간 구조를 지키려고 앞에서 자르지 않고 균등 표본을 쓴다
    i_fit = np.sort(rng.choice(i_tr, SUB, replace=False))
    tag = f"부분표본 {SUB:,}"
else:
    i_fit = i_tr
    tag = f"전체 {len(i_tr):,}"
H(f"학습 {tag} -> 관문 2024 {len(i_gt):,}")

from pytabkit import MLP_PLR_D_Classifier, RealMLP_TD_Classifier

MODELS = {
    "MLP-PLR (TD)": lambda s: MLP_PLR_D_Classifier(
        random_state=s, n_threads=4, verbosity=0, device="cpu"),
}
SEEDS = (1,)
res = {}
for name, mk in MODELS.items():
    ps, t0 = [], time.time()
    try:
        for s in SEEDS:
            m = mk(s)
            m.fit(X[i_fit], y[i_fit].astype(int), cat_col_names=cat_idx)
            ps.append(m.predict_proba(X[i_gt])[:, 1])
            print(f"  {name} seed{s} 학습완료 {time.time()-t0:.0f}s", flush=True)
    except TypeError:
        # cat_col_names 를 안 받는 버전 대응 — 전부 수치로 넣는다
        ps, t0 = [], time.time()
        for s in SEEDS:
            m = mk(s)
            m.fit(X[i_fit], y[i_fit].astype(int))
            ps.append(m.predict_proba(X[i_gt])[:, 1])
            print(f"  {name} seed{s} 학습완료(범주지정 없이) {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"  {name} 실패: {type(e).__name__} {e}")
        continue
    p = np.mean(ps, 0)
    res[name] = p
    line = "  ".join(f"shift{c:+.2f} {bss(sh(p, c)[~fv], yv[~fv]):7.1f}" for c in (0.0, -0.05))
    print(f"  {name:16s} 관문 1군  {line}   ({time.time()-t0:.0f}s)", flush=True)
    np.save(os.path.join(SC, f"pytk_{name.split()[0].replace('-','_')}_pred.npy"), p)

H("비교")
print("  같은 관문 (1군, shift 0 / −0.05)")
print("    FC 손구현 3시드     752.2 / 785.5")
print("    sklearn 전체단독    814.2 / 856.2")
print("    CatBoost 60+40      850.6 / 886.3")
for n, p in res.items():
    print(f"    {n:18s} {bss(sh(p,0.0)[~fv], yv[~fv]):7.1f} / "
          f"{bss(sh(p,-0.05)[~fv], yv[~fv]):7.1f}")
