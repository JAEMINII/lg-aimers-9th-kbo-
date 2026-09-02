# -*- coding: utf-8 -*-
"""
FC 와 GBDT 의 오차 상관을 재고 최적 앙상블 비율을 찾는다.

왜 상관이 핵심인가
  FC 단독은 관문 1군 785.5 로 CatBoost 60+40(886.3)보다 100점 낮다.
  그런데 앙상블 이득은 단독 점수가 아니라 '틀리는 지점이 다른가' 로 정해진다.

  근거: FC 시드 3개의 단독 점수가 506.8 / 545.8 / 712.1 로 편차가 극심한데
        3시드 앙상블은 785.5 다. 단일 평균(588.2)보다 +197.
        시드끼리도 이만큼 상관이 낮다는 뜻이고, 알고리즘이 다른 GBDT 와는 더 낮을 것이다.

  반대로 상관이 0.95 이상이면 FC 가 GBDT 와 같은 일을 한 것이라 섞을 값어치가 없다.

앞서 LightGBM+CatBoost 앙상블이 관문에서 단독보다 낮았던 이유도 여기 있다.
둘 다 GBDT 라 상관이 높았다. 근사 방식이 다른 모델이라야 이득이 난다.
"""
import os, json
import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
ROOT = r"c:\Users\jaemin.DESKTOP-2B30D1D\Desktop\재민\재민공부\aimers_재민"
D = os.path.join(ROOT, "open (1)", "data")
TGT = "control_success"

# 관문 정답과 game_type (fc_proper 와 같은 행 순서)
tr = pd.read_csv(os.path.join(D, "train.csv"), encoding="utf-8-sig",
                 usecols=["row_id", "season", "game_type", TGT])
tr["_r"] = tr.row_id.str.slice(6).astype("int32")
tr = tr.sort_values("_r").reset_index(drop=True)
m_va = (tr.season == 2024).values
yv = tr.loc[m_va, TGT].to_numpy(dtype=np.float64)
fv = (tr.loc[m_va, "game_type"].values == "F")
print(f"관문 {len(yv):,}행  (R {int((~fv).sum()):,} / F {int(fv.sum()):,})")

def bss(p, y):
    r = y.mean()
    return 100000 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))
def sh(p, c):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + c)))

fc = np.load(os.path.join(SC, "fc_gate_pred.npy"))
bl = json.load(open(os.path.join(SC, "blend_state.json"), encoding="utf-8"))
def blend(tag, w=0.6):
    d = bl[tag]
    full = np.asarray(d["full"]); r = np.asarray(d["r"]); f = np.asarray(d["f"])
    return w * full + (1 - w) * np.where(fv, f, r)
cb = blend("CatBoost")
sk = blend("sklearn")
print(f"FC {fc.shape}  CatBoost {cb.shape}  sklearn {sk.shape}")

print("\n" + "=" * 78)
print("1. 예측·오차 상관 (1군 행만)")
print("=" * 78)
m = ~fv
P = {"FC": fc[m], "CatBoost": cb[m], "sklearn": sk[m]}
E = {k: v - yv[m] for k, v in P.items()}
names = list(P)
print(f"  {'':12s}" + "".join(f"{n:>12}" for n in names))
for a in names:
    print(f"  예측 {a:8s}" + "".join(f"{np.corrcoef(P[a], P[b])[0,1]:12.4f}" for b in names))
print()
for a in names:
    print(f"  오차 {a:8s}" + "".join(f"{np.corrcoef(E[a], E[b])[0,1]:12.4f}" for b in names))

print("\n" + "=" * 78)
print("2. FC x CatBoost 비율 스윕 (1군 채점)")
print("=" * 78)
print(f"  {'FC 비중':>8}{'shift 0':>12}{'shift −0.05':>14}")
best = (None, -1e9, None)
for w in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.0):
    row = []
    for c in (0.0, -0.05):
        q = sh(w * fc + (1 - w) * cb, c)
        s = bss(q[m], yv[m])
        row.append(s)
        if s > best[1]:
            best = (w, s, c)
    print(f"  {w:8.2f}{row[0]:12.1f}{row[1]:14.1f}")
print(f"\n  최고: FC {best[0]:.2f} + CatBoost {1-best[0]:.2f}, shift {best[2]:+.2f}  -> {best[1]:.1f}")
print(f"  CatBoost 단독 최고 = {max(bss(sh(cb,c)[m], yv[m]) for c in (0.0,-0.05)):.1f}")

print("\n" + "=" * 78)
print("3. 3종 (FC + CatBoost + sklearn)")
print("=" * 78)
for wf in (0.10, 0.15, 0.20, 0.25):
    rest = 1 - wf
    q0 = sh(wf * fc + rest * 0.5 * cb + rest * 0.5 * sk, 0.0)
    q5 = sh(wf * fc + rest * 0.5 * cb + rest * 0.5 * sk, -0.05)
    print(f"  FC {wf:.2f} + CB {rest*0.5:.2f} + SK {rest*0.5:.2f}   "
          f"shift0 {bss(q0[m], yv[m]):8.1f}   shift−.05 {bss(q5[m], yv[m]):8.1f}")
