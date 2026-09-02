# -*- coding: utf-8 -*-
"""13투 잔차를 시즌누적 투구수(cur_n) 분위로 분해 — 2024 폴드 위상창(3~7월)."""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
def P(s): sys.stdout.buffer.write((s+"\n").encode("utf-8"))
DL="colab/_dl"; DATA="open (1)/data"
W4=np.array([0.29,0.12,0.29,0.30]); CW=np.array([0.45,0.10,0.25,0.20])
def bss(p,t):
    r=t.mean(); return 100000*(1-((p-t)**2).mean()/(r*(1-r)))
def LG(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def IV(z): return 1/(1+np.exp(-z))
def ld(*n):
    a=[np.load(os.path.join(DL,x)).astype(np.float64) for x in n]
    a=[x.mean(0) if x.ndim==2 else x for x in a]; return np.mean(a,0)
tr=pd.read_csv(os.path.join(DATA,"train.csv"),encoding="utf-8-sig",
    usecols=["row_id","season","game_month","game_type","pitcher_id","batter_id",
             "pitcher_team_id","batter_team_id","control_success","asof_pitcher_n"])
rid=tr["row_id"].str.slice(6).astype("int64")
tr=tr.assign(_r=rid).sort_values("_r").reset_index(drop=True)
season=tr["season"].to_numpy()
va_m=season==2024
va=tr[va_m]; trn=tr[season<2024]
yv=va["control_success"].to_numpy(float)
isf=va["game_type"].astype(str).to_numpy()=="F"
pidv=va["pitcher_id"].to_numpy(); bidv=va["batter_id"].to_numpy()
wp=set(trn["pitcher_id"]); wb=set(trn["batter_id"])
lp=trn.groupby("pitcher_id")["season"].max()
cb=ld("ta2024_base.npy")
tab=(0.6*np.load(f"{DL}/c4g_2024_c4_all_s42.npy")+0.4*np.load(f"{DL}/c4g_2024_c4_regular_s42.npy")).astype(float)
din=ld("dg_2024_DIN.npy"); ms6=ld("msr_state6_s42.npy","msr_state6_s1.npy")
mso=ld("msp_2024_ms_s2_s42.npy","msp_2024_ms_s2_s1.npy"); mif=ld("msif_2024_s42.npy","msif_2024_s1.npy")
n=len(yv)
cold=np.array([p not in wp for p in pidv])|np.array([b not in wb for b in bidv])
lsp=np.array([lp.get(v,-1) for v in pidv])
st=~cold&(lsp<=2022)&~isf; idf=(cold&~isf)|st
r=np.where(idf,mif,np.where(isf,mso,ms6))
F=np.column_stack([cb,tab,din,r]); Wm=np.tile(W4,(n,1)); Wm[cold]=CW
z=LG((F*Wm).sum(1))
m=~isf
r0=minimize_scalar(lambda c:-bss(IV(z+c)[m],yv[m]),bounds=(-0.3,0.3),method="bounded")
pz=IV(z+r0.x)
# cur_n = asof_n - 학습측 총계
tot=trn.groupby("pitcher_id")["asof_pitcher_n"].max()
nall=va["asof_pitcher_n"].fillna(0).to_numpy(float)
n0=np.array([tot.get(p,0.0) for p in pidv])
curn=np.clip(nall-n0,0,None)
pt=va.pitcher_team_id.to_numpy()
mo=va.game_month.to_numpy()
mask=(pt==13)&m&(mo>=3)&(mo<=7)
P("== 13투 R행 (3~7월, 2024폴드) cur_n 3분위별 잔차")
q=np.quantile(curn[mask],[0.33,0.67])
P(f"  분위 경계 cur_n: {q.round(0)}")
for tag,mm in (("low",mask&(curn<=q[0])),("mid",mask&(curn>q[0])&(curn<=q[1])),("high",mask&(curn>q[1]))):
    P(f"  {tag}  n={int(mm.sum())}  잔차 {100*(yv[mm].mean()-pz[mm].mean()):+.2f}pp")
P("== 대조: 비13 같은 분해")
mask2=(pt!=13)&(va.batter_team_id.to_numpy()!=13)&m&(mo>=3)&(mo<=7)
for tag,mm in (("low",mask2&(curn<=q[0])),("mid",mask2&(curn>q[0])&(curn<=q[1])),("high",mask2&(curn>q[1]))):
    P(f"  {tag}  n={int(mm.sum())}  잔차 {100*(yv[mm].mean()-pz[mm].mean()):+.2f}pp")
