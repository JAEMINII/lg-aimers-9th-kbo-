# -*- coding: utf-8 -*-
"""plat_dev 의 **사전값**을 as-of 로 바꾼다. 고침이 절반만 되어 있었다.

무엇이 남아 있었나
    창 의존 이진 검사에서 features44 의 plat_dev 가 여전히 걸렸다 (최대차 7.9e-03).
    지인 판(4.69e-02)보다 6배 작지만 0 이 아니다.

    구조를 보면 이유가 명확하다.
        PCC = PC.groupby(level=0).cumsum().groupby(level=0).shift(1)
              -> (투수, 시즌, 타자손) 누적. 시즌 단위로 **이미 as-of** 다.
        lgph = hist.groupby([투수손, 타자손])[y].mean()   <- 학습창 전체
        lgp  = hist.groupby(투수손)[y].mean()             <- 학습창 전체
        gm_  = hist[y].mean()                             <- 학습창 전체
    누적은 as-of 인데 **사전값이 창 전체**다. ALPHA_PLAT=300 으로 섞이므로
    표본이 적은 투수·시즌일수록 사전값 비중이 크다. 2019 행은 PCC 가 0 이라
    plat_dev 가 사실상 (lgph - lgp) 그 자체다.

고치는 법
    행의 시즌 S 에 대해 **season < S** 인 행들로만 리그값을 만든다.
    S=2019 는 앞선 시즌이 없으므로 사전값을 못 만든다 -> plat_dev = 0 (중립).
    그게 정직한 as-of 답이다. 그 해엔 아직 정보가 없다.

진단
    누출판과 as-of 판의 표적 상관을 비교한다. plat_dev 전체 고침 때
    +0.02368 -> +0.01478 이었고 그게 LB +11 이었다. 남은 몫을 여기서 잰다.
"""
import gc
import os
import sys

import numpy as np
import pandas as pd

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
ROOT = os.path.dirname(SC)
DATA = os.path.join(ROOT, "open (1)", "data")
DL = os.path.join(SC, "_dl")
ALPHA_PLAT = 300.0
TGT = "control_success"

import features44 as F                                          # noqa: E402

d = F.build(DATA, VS=2025, return_frame=True)
fr = d["frame"]
y = d["y"].astype(np.float64)
season = d["season"].astype(int)
old = d["X44"][:, list(d["F44"]).index("plat_dev")].astype(np.float64)
print(f"  {len(fr):,}행   시즌 {season.min()}~{season.max()}")

raw = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                  usecols=["row_id", "pitcher_id", "pitcher_hand", "batter_hand",
                           "season", TGT])
raw = raw.set_index("row_id").reindex(fr["row_id"].to_numpy()).reset_index()
ph_row = raw["pitcher_hand"].to_numpy()
bh_row = raw["batter_hand"].to_numpy()
pid = raw["pitcher_id"].to_numpy()

# ---- (투수, 시즌, 타자손) 누적. features44 와 같은 규약
PC = raw.groupby(["pitcher_id", "season", "batter_hand"])[TGT] \
        .agg(["size", "sum"]).unstack(fill_value=0)
PC.columns = [f"{x}{int(h)}" for x, h in PC.columns]
for h in (1, 2):
    for x in ("size", "sum"):
        if f"{x}{h}" not in PC.columns:
            PC[f"{x}{h}"] = 0
PCC = PC.groupby(level=0).cumsum().groupby(level=0).shift(1).fillna(0.)
PCC["n_all"] = PCC.size1 + PCC.size2
PCC["s_all"] = PCC.sum1 + PCC.sum2
PH = raw.groupby("pitcher_id").pitcher_hand.first()
gc.collect()

# ---- 시즌별 as-of 리그값 (season < S 인 행만)
seasons = sorted(raw["season"].unique())
LGPH, LGP = {}, {}
for S in seasons:
    prev = raw[raw.season < S]
    if len(prev) < 1000:
        LGPH[S], LGP[S] = None, None
        continue
    LGPH[S] = prev.groupby(["pitcher_hand", "batter_hand"])[TGT].mean().to_dict()
    LGP[S] = prev.groupby("pitcher_hand")[TGT].mean().to_dict()
    print(f"  {S}  사전값 표본 {len(prev):,}행   "
          f"리그 {prev[TGT].mean():.4f}")

idx = pd.MultiIndex.from_arrays([pid, raw["season"].to_numpy()])
sub = PCC.reindex(idx)
n_all = sub["n_all"].to_numpy(np.float64)
s_all = sub["s_all"].to_numpy(np.float64)
ph_of = PH.reindex(pid).to_numpy()

new = np.zeros(len(raw), np.float64)
for S in seasons:
    m = raw["season"].to_numpy() == S
    if LGPH[S] is None:            # 앞선 시즌이 없다 -> 중립 0
        continue
    prh = np.array([LGPH[S].get((p_, int(h_)), np.nan)
                    for p_, h_ in zip(ph_of[m], bh_row[m])])
    pra = np.array([LGP[S].get(p_, np.nan) for p_ in ph_of[m]])
    sz = np.where(bh_row[m] == 1, sub["size1"].to_numpy()[m],
                  sub["size2"].to_numpy()[m]).astype(np.float64)
    sm = np.where(bh_row[m] == 1, sub["sum1"].to_numpy()[m],
                  sub["sum2"].to_numpy()[m]).astype(np.float64)
    ph_val = (sm + ALPHA_PLAT * prh) / (sz + ALPHA_PLAT)
    pa_val = (s_all[m] + ALPHA_PLAT * pra) / (n_all[m] + ALPHA_PLAT)
    new[m] = np.nan_to_num(ph_val - pa_val)

print(f"\n  {'':10s} {'표적상관':>12s}")
mm = np.isfinite(old) & np.isfinite(new)
print(f"  누출판     {np.corrcoef(old[mm], y[mm])[0,1]:+12.5f}")
print(f"  as-of 판   {np.corrcoef(new[mm], y[mm])[0,1]:+12.5f}")
print(f"  두 판 상관 {np.corrcoef(old[mm], new[mm])[0,1]:12.4f}   "
      f"최대차 {np.abs(old-new).max():.4e}")
print("  참고 — 전체 고침 때는 +0.02368 -> +0.01478 (LB +11) 이었다.")
print("  2019 는 앞선 시즌이 없어 0 이다. 전체의 "
      f"{(season==2019).mean()*100:.1f}% 다.")

out = pd.DataFrame({"row_id": fr["row_id"].to_numpy(),
                    "plat_dev_asof": new.astype(np.float32)})
out.to_csv(os.path.join(DL, "plat_asof.csv.gz"), index=False)
print(f"\n  저장  colab/_dl/plat_asof.csv.gz")
