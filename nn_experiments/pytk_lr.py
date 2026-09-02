# -*- coding: utf-8 -*-
"""
전체 122만 행에서 배치·학습률을 통제해 붕괴를 고친다.

지금까지 잰 조합
                              배치        조기종료   데이터     판별력
  기본값                      기본(가변)    켬        20만        607.1
  기본값                      기본(가변)    켬       122만         98.5   <- 붕괴
  배치512 고정                512          켬        20만        519.8
  조기종료 무력 ep8           512          끔        20만        −88.0
  조기종료 무력 ep30          512          끔        20만      −7153.5   <- 발산

배운 것
  1) 조기종료·체크포인트 선택은 반드시 켠다. 끄면 발산한다.
     체크포인트가 발산 전 지점을 건져내는 안전장치였다.
  2) 시간이 6배 데이터에 1.16배였다 -> 배치가 데이터에 비례해 커진 정황.
     리서치: "배치가 커지면 쓸 수 있는 학습률 범위가 크게 좁아진다"
     그러면 배치 6배에 학습률 그대로 = 범위를 벗어난 것이다.

이 실험
  전체 122만 행에서 배치를 512 로 고정하고, 학습률을 낮은 쪽으로 훑는다.
  조기종료는 켠 채로 둔다(기본값). 배치 512 라면 20만 행 384초 기준으로
  122만 행은 약 38분. 세 값이면 2시간이다.

  관문(2024)을 반복해서 보게 되므로 이 단계는 '튜닝' 이다.
  여기서 고른 설정은 리더보드로만 최종 확인할 수 있다. 그 한계를 알고 쓴다.

기준선 (같은 관문, 1군 채점)
  CatBoost 60+40   850.6 / 886.3     <- 넘어야 850
  MLP 최고         607.1 (20만 행)
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

i_fit = np.where(season < 2024)[0]
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

from pytabkit import MLP_PLR_D_Classifier as M

# 배치를 512 로 고정하고 학습률만 바꾼다.
# lr=None 은 pytabkit 내부 기본값. 그게 배치 512 에 맞는지 먼저 보고,
# 아니면 낮은 쪽으로 내려간다. tabular DL 통상 범위는 1e-4 ~ 5e-3.
# 1차 결과 (전체 122만 행, 판별력만)
#   배치512 lr기본   455.3   (붕괴판본 98.5 대비 +357 -> 배치가 원인의 일부 확정)
#   배치512 lr 4e-4  −80.8   (낮추니 더 나쁘다 -> 학습률이 높아서 문제가 아니었다)
# lr 1e-4 는 더 나쁠 게 뻔해서 취소했다.
#
# 남은 가설: 조기종료가 122만 행에서 너무 일찍 끊어 미학습으로 끝난다.
#   es_patience 를 늘리되 use_checkpoints 는 켠 채로 둔다.
#   (A2 에서 체크포인트를 끄면 ep30 에 −7,153 으로 발산했다. 안전장치는 유지)
CASES = [
    ("배치512 patience32", dict(batch_size=512, es_patience=32)),
    ("배치256 lr기본",     dict(batch_size=256)),
]
print(f"전체 {len(i_fit):,} 행 학습 -> 관문 2024 {len(i_gt):,}")
print(f"기준: CatBoost 60+40 = 850.6 / 886.3   ·   MLP 최고 607.1 (20만 행)")
print(f"붕괴판본(기본배치 122만행) = 15.3 / 197.2, 판별력 98.5\n")
PROG = os.path.join(SC, "scratchpad", "lr_progress.txt")
for name, kw in CASES:
    t0 = time.time()
    try:
        mm = M(random_state=1, n_threads=4, verbosity=0, device="cpu", **kw)
        mm.fit(X[i_fit], y[i_fit].astype(int), cat_col_names=cat_idx)
        p = mm.predict_proba(X[i_gt])[:, 1]
    except Exception as e:
        line = f"  {name:18s} 실패: {type(e).__name__} {str(e)[:100]}"
        print(line, flush=True)
        open(PROG, "a", encoding="utf-8").write(line + "\n")
        continue
    np.save(os.path.join(SC, f"lr_{name.split()[-1]}.npy"), p)
    line = (f"  {name:18s} shift0 {bss(sh(p,0.0)[m], yv[m]):8.1f}  "
            f"shift−.05 {bss(sh(p,-0.05)[m], yv[m]):8.1f}  "
            f"판별력만 {bss(recentered(p)[m], yv[m]):8.1f}   "
            f"예측평균 {p[m].mean():.4f}(실제 0.4897)   ({time.time()-t0:.0f}s)")
    print(line, flush=True)
    open(PROG, "a", encoding="utf-8").write(line + "\n")
print("\n판독")
print("  600 대가 나오면   배치가 원인 확정. 학습률을 더 훑어 850 을 노린다")
print("  여전히 100 근처   배치가 아님. 구조·전처리 쪽을 다시 봐야 함")
