# -*- coding: utf-8 -*-
"""tabm_gate.ipynb 생성. 셀 내용을 여기서 관리한다."""
import io
import json

md = lambda s: {"cell_type": "markdown", "metadata": {},
                "source": s.strip("\n").split("\n")}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n").split("\n")}

C = []

C.append(md("""
# TabM 관문 실험

지인 결과(리더보드 1012)를 **관문에서 재현**하고, 우리 CatBoost 와 섞으면
얼마나 오르는지 재는 노트북이다. 제출본은 여기서 만들지 않는다.

## 관문이 리더보드로 환산된다

    리더보드 ~= 0.97 x 관문 + 146       (5회 측정, 3회 연속 적중)

    우리 위치   CatBoost(시즌가중2.0) 단독   관문 906.7
                + MLP 0.20                  관문 912.6  ->  리더보드 1021 (실측)
    지인 TabM   리더보드 1012              ->  관문 환산 약 893

## 왜 로컬이 아니라 Colab 인가

CPU 로 브랜치당 2.7시간, 3브랜치 8시간이다. T4 면 10~25분이면 끝난다.
로컬에서 TabM 을 잘못된 설정으로 돌려 3시간 43분을 태운 적이 있다.

## 필요한 파일

    aimers/
      open/data/train.csv          368 MB
      open/data/test.csv
      colab/features44.py          44피처 생성 (로컬 검증본)
      colab/tabm_model.py          TabM 구현
      colab/cb_gate.npy            우리 CatBoost 관문 예측
      colab/mlp_gate.npy           우리 MLP 관문 예측

`ROOT` 만 본인 경로로 바꾸면 된다.
"""))

C.append(code("""
ROOT = "/content/drive/MyDrive/aimers"          # <- 본인 경로

from google.colab import drive
drive.mount("/content/drive")

import sys, os, time, gc
sys.path.insert(0, ROOT + "/colab")
import numpy as np, torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "|", DEV,
      "|", torch.cuda.get_device_name(0) if DEV == "cuda" else "")
assert DEV == "cuda", "런타임 유형을 GPU 로 바꾸세요 (런타임 > 런타임 유형 변경)"
"""))

C.append(md("""
## 1. 44피처

로컬에서 검증된 코드다. 지인 SPEC 의 44피처와 **순서까지 동일**함을 확인했다.

`VS=2024` 면 2019~2023 이 학습 구간이고, 룩업 테이블(리그 평균, 카운트 효과,
플래툰)도 그 구간에서만 만들어진다. 선수 이력은 cumsum -> shift(1) 이라
자기 자신이 안 들어간다. 이 규약을 어기면 관문 점수가 통째로 거짓이 된다.
"""))

C.append(code("""
import features44 as F
import tabm_model as T

t0 = time.time()
d = F.build(ROOT + "/open/data", VS=2024)
X, y = d["X44"], d["y"]
m_tr, m_va, is_f, season = d["m_tr"], d["m_va"], d["is_f"], d["season"]
print(f"{time.time()-t0:.0f}s   X {X.shape}   학습 {m_tr.sum():,}   검증 {m_va.sum():,}")

Xn, Xc, cards = T.prep(X, m_tr, d["cat_idx"])
print(f"수치 {Xn.shape[1]}열 (35 + 결측표시)   범주 {Xc.shape[1]}열   카디널리티 {cards.tolist()}")

gate = np.where(m_va)[0]
mm   = ~is_f[gate]                      # 채점은 1군만
yv   = y[gate].astype(np.float64)
print(f"관문 1군 {mm.sum():,}행   실제 성공률 {yv[mm].mean():.4f}")
"""))

C.append(md("""
## 2. 학습 함수

지인 SPEC 그대로다.

| | 값 |
|---|---|
| k / n_blocks / d_block / dropout | 32 / 3 / 256 / 0.1 |
| 수치 임베딩 | LinearReLU, d=16 |
| batch / lr / weight_decay / clip | 2048 / 2e-3 / 3e-4 / 5.0 |
| scheduler | cosine |
| loss | 0.5 x BCE + 0.5 x Brier |
| Stage 1 | 학습 구간 전체, 2 epoch |
| Stage 2 | **최근 시즌만**, 1 epoch, lr 2e-4, 마지막 block + head 만 |

**Stage 2 가 핵심이다.** 우리는 학습 구간을 '전체 vs 한 시즌만' 이분법으로만
봤는데, 이건 전체로 표현을 배우고 최근 시즌으로 출력만 맞추는 제3의 방식이다.
"""))

C.append(code("""
def run_branch(idx, recent_season, stage2=True, ep1=2, ep2=1,
               lr1=2e-3, lr2=2e-4, k=32, seed=42, tag=""):
    torch.manual_seed(seed); np.random.seed(seed)
    m = T.TabM(Xn.shape[1], cards, k=k, n_blocks=3, d_block=256,
               dropout=0.1, d_embedding=16).to(DEV)
    t0 = time.time()
    print(f"  [{tag}] Stage1  {len(idx):,}행")
    T.train_stage(m, Xn, Xc, y, idx, epochs=ep1, lr=lr1, seed=seed)
    if stage2:
        s2 = idx[season[idx] == recent_season]
        print(f"  [{tag}] Stage2  {len(s2):,}행  (마지막 block + head 만)")
        T.train_stage(m, Xn, Xc, y, s2, epochs=ep2, lr=lr2,
                      params=m.last_block_params(), seed=seed)
    p = T.predict(m, Xn, Xc, gate)
    print(f"  [{tag}] {time.time()-t0:.0f}s")
    del m; gc.collect(); torch.cuda.empty_cache()
    return p


def run_all(recent_season=2023, **kw):
    \"\"\"all / futures / regular 3브랜치를 0.6:0.4 로 합친다 (우리 라우팅과 동일).\"\"\"
    tr_idx = np.where(m_tr)[0]
    P = {}
    for nm, idx in (("all", tr_idx),
                    ("futures", tr_idx[is_f[tr_idx]]),
                    ("regular", tr_idx[~is_f[tr_idx]])):
        P[nm] = run_branch(idx, recent_season, tag=nm, **kw)
    return np.where(is_f[gate], 0.6*P["all"] + 0.4*P["futures"],
                                0.6*P["all"] + 0.4*P["regular"]), P
"""))

C.append(md("""
## 3. 기본 구성 학습

관문에서는 '최근 시즌' 이 2023 이다 (제출 시에는 2024).
"""))

C.append(code("""
tabm_p, parts = run_all(recent_season=2023)
np.save(ROOT + "/colab/tabm_gate.npy", tabm_p)

s, c = F.best_shift(tabm_p[mm], yv[mm])
print(f"\\nTabM 단독 판별력 {s:.1f}  (최적시프트 {c:+.4f})")
print(f"  shift 0 기준     {F.bss(tabm_p[mm], yv[mm]):.1f}")
print(f"  예측 평균 {tabm_p[mm].mean():.4f}  실제 {yv[mm].mean():.4f}")
"""))

C.append(md("""
## 4. 우리 CatBoost 와 합치기

**보는 것은 단독 점수가 아니라 상관이다.** 오늘 확인된 경계가 뚜렷하다.

    CatBoost <-> MLP            0.907   ->  +5.9
    CatBoost <-> LGB/HistGB     0.962   ->  +0.5
    CatBoost <-> 같은 CatBoost   0.988   ->  +0.5

다만 상관이 낮은 것만으로는 부족하다. RealMLP 는 상관 0.78 인데 단독이 515 라
이득이 +1.0 이었다. **단독 성능과 낮은 상관이 동시에** 필요하다.
"""))

C.append(code("""
cb  = np.load(ROOT + "/colab/cb_gate.npy")
mlp = np.load(ROOT + "/colab/mlp_gate.npy")
cur = 0.8*cb + 0.2*mlp                      # 현 제출본 = 리더보드 1021

for nm, p in (("CatBoost 단독", cb), ("MLP 단독", mlp),
              ("현 제출본(0.8CB+0.2MLP)", cur), ("TabM 단독", tabm_p)):
    print(f"  {nm:<24s} {F.best_shift(p[mm], yv[mm])[0]:8.1f}")
print(f"\\n  TabM <-> CatBoost 상관 {np.corrcoef(tabm_p[mm], cb[mm])[0,1]:.4f}")
print(f"  TabM <-> MLP      상관 {np.corrcoef(tabm_p[mm], mlp[mm])[0,1]:.4f}")

base = F.best_shift(cur[mm], yv[mm])[0]
print(f"\\n현 제출본에 TabM 을 얹으면 (기준 {base:.1f} = 리더보드 1021)")
print(f"  {'TabM비중':>9s}{'판별력':>10s}{'대비':>9s}{'리더보드 예상':>14s}")
for w in np.arange(0, 0.85, 0.05):
    s = F.best_shift(((1-w)*cur + w*tabm_p)[mm], yv[mm])[0]
    print(f"  {w:9.2f}{s:10.1f}{s-base:+9.1f}{1021 + 0.97*(s-base):14.0f}")

print(f"\\nMLP 를 TabM 으로 교체하면")
for w in np.arange(0.1, 0.75, 0.05):
    s = F.best_shift(((1-w)*cb + w*tabm_p)[mm], yv[mm])[0]
    print(f"  {'CB '+format(1-w,'.2f')+' + TabM '+format(w,'.2f'):<22s}{s:9.1f}"
          f"{s-base:+9.1f}{1021 + 0.97*(s-base):14.0f}")
"""))

C.append(md("""
## 5. 변형 실험

**한 번에 하나씩만 바꾼다.** 두 개를 같이 바꾸면 원인 분리가 안 된다.

지인은 시즌 가중치를 안 쓰고 Stage2 로 최근 시즌에 맞춘다. 우리는 시즌 가중치
(관문 +22)를 쓴다. 둘은 같은 일을 다르게 하는 것이라 합치면 과보정일 수도,
상보적일 수도 있다. 재봐야 안다.
"""))

C.append(code("""
VARIANTS = {
    "Stage2 없음":        dict(stage2=False),
    "Stage1 4 epoch":     dict(ep1=4),
    "Stage2 2 epoch":     dict(ep2=2),
    "Stage2 lr 5e-4":     dict(lr2=5e-4),
    "k=64":               dict(k=64),
}
res = {"기본": tabm_p}
for nm, kw in VARIANTS.items():
    print(f"\\n=== {nm}")
    res[nm], _ = run_all(recent_season=2023, **kw)
    np.save(ROOT + f"/colab/tabm_{nm.replace(' ', '_')}.npy", res[nm])

print(f"\\n  {'구성':<18s}{'단독':>9s}{'CB상관':>9s}{'최적비중':>9s}{'섞은뒤':>9s}{'LB예상':>9s}")
for nm, p in res.items():
    bw, bs_ = 0.0, base
    for w in np.arange(0.05, 0.85, 0.05):
        s = F.best_shift(((1-w)*cur + w*p)[mm], yv[mm])[0]
        if s > bs_: bs_, bw = s, w
    print(f"  {nm:<18s}{F.best_shift(p[mm], yv[mm])[0]:9.1f}"
          f"{np.corrcoef(p[mm], cb[mm])[0,1]:9.4f}{bw:9.2f}{bs_:9.1f}"
          f"{1021 + 0.97*(bs_-base):9.0f}")
"""))

C.append(md("""
## 6. 다음 단계

관문에서 이득이 확인되면 제출본을 만든다. 그때 바뀌는 것은 두 가지뿐이다.

1. `VS=2025` 로 피처를 만들고 (학습 구간이 2019~2024 가 된다)
2. `recent_season=2024` 로 Stage2 를 돌린다

제출은 **numpy 추론**이라 torch 가 필요 없다. 지인도 npz 3개로 제출했다.
TabM 은 BatchEnsemble 이라 어댑터가 원소곱이고, 우리가 MLP 에서 뚫어놓은
추출 경로(분위수변환 -> 임베딩 -> 행렬곱)에 곱셈 두 줄만 더하면 된다.

`tabm_gate.npy` 와 변형들을 드라이브에 저장해두면 로컬에서 이어서 분석할 수 있다.
"""))

nb = {"cells": C,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"}},
      "nbformat": 4, "nbformat_minor": 0}

out = io.open("tabm_gate.ipynb", "w", encoding="utf-8")
json.dump(nb, out, ensure_ascii=False, indent=1)
out.close()
print("생성: tabm_gate.ipynb   셀", len(C), "개")
