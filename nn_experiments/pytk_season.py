# -*- coding: utf-8 -*-
"""
최근 시즌만 학습 — 데이터가 많을수록 나빠지는 현상의 진짜 원인을 시험한다.

지금까지 (전체 122만 행, 관문 2024 1군 채점, 판별력만)
    기본배치                     98.5     붕괴
    배치512                     455.3     배치 고정이 +357
    배치512 lr 4e-4             −80.8     학습률 낮추면 악화
    배치512 patience32          455.3     lr기본과 소수점까지 동일
                                          -> 조기종료는 애초에 안 걸렸다. 가설 배제
    (참고) 기본배치 20만 행       607.1     데이터가 적은데 더 좋다

남은 설명
  데이터가 많을수록 2019~2023 을 정밀하게 맞추고, 2024 는 리그 수준이 다르므로
  과거에 잘 맞을수록 미래에 더 빗나간다. 버그가 아니라 실제 현상일 수 있다.

  20만 행 표본은 2019~2023 에서 '균등' 추출이라 분포는 같고 양만 적었다.
  양이 적어 과거에 덜 과적합했고 그래서 2024 에 더 맞았다.

  그렇다면 줄이는 방향이 중요하다:
      무작위로 줄이기   옛 데이터도 그대로 섞임
      최근만 쓰기       분포가 2024 에 가까워짐          <- 이걸 시험한다

리그 성공률 추이 (드리프트)
    2019 54.4% / 2020 54.0% / 2021 53.6% / 2022 53.2% / 2023 51.8% / 2024 49.9%
  2019 와 2024 는 4.5%p 차이다. 2019 데이터가 2024 예측에 도움이 되는지 자체가 의문이다.

이 시험은 GBDT 에도 적용해볼 값어치가 있다. 다만 GBDT 는 season 을 피처로 갖고
드리프트 보정도 하고 있어 이득이 작을 수 있다. 먼저 MLP 에서 효과를 본다.
"""
import os, sys, time
import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(SC, "nn_cache.npz"), allow_pickle=False)
Xc, Xn, y, season, is_f = z["Xc"], z["Xn"], z["y"], z["season"], z["is_f"]
X = np.concatenate([Xc.astype(np.float32), Xn], 1)
cat_idx = list(range(Xc.shape[1]))
del Xc, Xn
z.close()
import gc; gc.collect()

i_gt = np.where(season == 2024)[0]
fv, yv = is_f[i_gt], y[i_gt]
m = ~fv

def bss(p, yy):
    r = yy.mean()
    return 100000 * (1 - ((p - yy) ** 2).mean() / (r * (1 - r)))
def sh(p, c):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + c)))
def recentered(p, tgt=0.4897):
    lp = np.log(np.clip(p, 1e-6, 1-1e-6) / (1 - np.clip(p, 1e-6, 1-1e-6)))
    C = np.log(tgt / (1 - tgt))
    return 1 / (1 + np.exp(-(lp - lp.mean() + C)))

print("시즌별 행수와 성공률")
for s in range(2019, 2025):
    k = season == s
    print(f"  {s}  {int(k.sum()):>9,}행   성공률 {y[k].mean()*100:5.2f}%")

from pytabkit import MLP_PLR_D_Classifier as M

# 배치 512 고정(확인된 개선), 나머지 기본값. 학습 구간만 바꾼다.
CASES = [
    ("2023만",       [2023]),
    ("2022~2023",    [2022, 2023]),
    ("2021~2023",    [2021, 2022, 2023]),
]
print(f"\n관문 2024 {len(i_gt):,}행   비교 기준")
print(f"  2019~2023 전체(122만)  455.3      <- 지금까지 전체 데이터 최고")
print(f"  무작위 20만 (기본배치)   607.1      <- 데이터가 적은데 더 좋았다")
print(f"  CatBoost 60+40         886.3      <- 넘어야 할 선\n")
PROG = os.path.join(SC, "scratchpad", "season_progress.txt")
for name, yrs in CASES:
    i_fit = np.where(np.isin(season, yrs))[0]
    t0 = time.time()
    try:
        mm = M(random_state=1, n_threads=4, verbosity=0, device="cpu", batch_size=512)
        mm.fit(X[i_fit], y[i_fit].astype(int), cat_col_names=cat_idx)
        p = mm.predict_proba(X[i_gt])[:, 1]
    except Exception as e:
        line = f"  {name:12s} 실패: {type(e).__name__} {str(e)[:100]}"
        print(line, flush=True); open(PROG, "a", encoding="utf-8").write(line + "\n")
        continue
    np.save(os.path.join(SC, f"season_{name.replace('~','_')}.npy"), p)
    line = (f"  {name:12s} {len(i_fit):>9,}행   shift0 {bss(sh(p,0.0)[m], yv[m]):8.1f}  "
            f"shift−.05 {bss(sh(p,-0.05)[m], yv[m]):8.1f}  "
            f"판별력만 {bss(recentered(p)[m], yv[m]):8.1f}   "
            f"예측평균 {p[m].mean():.4f}   ({time.time()-t0:.0f}s)")
    print(line, flush=True); open(PROG, "a", encoding="utf-8").write(line + "\n")
print("\n판독")
print("  최근만 쓴 쪽이 455 를 크게 넘으면  -> 옛 시즌이 해롭다. 학습 구간을 잘라야 한다")
print("  차이가 없으면                     -> 데이터 양이 아니라 다른 요인")
print("  최근만 쓴 쪽이 더 나쁘면           -> 데이터 양이 역시 중요. 과적합 가설 기각")
