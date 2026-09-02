# -*- coding: utf-8 -*-
"""마지막 3축: 13투 월별 재배분 / cold·stale 13투 부스트 / 불일치행 연화. 2024폴드."""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
def P(s): sys.stdout.buffer.write((s+"\n").encode("utf-8"))
DL="colab/_dl"; DATA="open (1)/data"
W4=np.array([0.3928,0.0600,0.1665,0.3808]); CW=np.array([0.45,0.10,0.25,0.20])
def bss(p,t):
    r=t.mean(); return 100000*(1-((p-t)**2).mean()/(r*(1-r)))
def LG(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def IV(z): return 1/(1+np.exp(-z))
def ld(*n):
    a=[np.load(os.path.join(DL,x)).astype(np.float64) for x in n]
    a=[x.mean(0) if x.ndim==2 else x for x in a]; return np.mean(a,0)
tr=pd.read_csv(os.path.join(DATA,"train.csv"),encoding="utf-8-sig",
    usecols=["row_id","season","game_month","game_type","pitcher_id","batter_id",
             "pitcher_team_id","batter_team_id","control_success"])
rid=tr["row_id"].str.slice(6).astype("int64")
tr=tr.assign(_r=rid).sort_values("_r").reset_index(drop=True)
y=tr["control_success"].to_numpy(float)
season=tr["season"].to_numpy()
va_m=season==2024
va=tr[va_m]; trn=tr[season<2024]
yv=y[va_m]
isf=va["game_type"].astype(str).to_numpy()=="F"
pidv=va["pitcher_id"].to_numpy(); bidv=va["batter_id"].to_numpy()
wp=set(trn["pitcher_id"]); wb=set(trn["batter_id"])
lp=trn.groupby("pitcher_id")["season"].max()
cb=ld("cbh4_2024_b58.npy")
tab=(0.6*np.load(f"{DL}/c4g_2024_c4_all_s42.npy")+0.4*np.load(f"{DL}/c4g_2024_c4_regular_s42.npy")).astype(float)
din=ld("dg_2024_DIN.npy"); ms6=ld("msr_state6_s42.npy","msr_state6_s1.npy")
mso=ld("msp_2024_ms_s2_s42.npy","msp_2024_ms_s2_s1.npy"); mif=ld("msif_2024_s42.npy","msif_2024_s1.npy")
n=len(yv)
coldp=np.array([p not in wp for p in pidv])
cold=coldp|np.array([b not in wb for b in bidv])
lsp=np.array([lp.get(v,-1) for v in pidv])
st=~cold&(lsp<=2022)&~isf; idf=(cold&~isf)|st
r=np.where(idf,mif,np.where(isf,mso,ms6))
Z4=np.column_stack([LG(cb),LG(tab),LG(din),LG(r)])
F=np.column_stack([cb,tab,din,r]); Wm=np.tile(W4,(n,1)); Wm[cold]=CW
z=LG((F*Wm).sum(1))
pt=va.pitcher_team_id.to_numpy(); bt=va.batter_team_id.to_numpy()
mo=va.game_month.to_numpy()
t13p=(pt==13)&~isf; t13b=(bt==13)&(pt!=13)&~isf
z=z+np.where(t13p,0.08,0)+np.where(t13b,0.06,0)
m=~isf
def sc(zz,mm):
    r_=minimize_scalar(lambda c:-bss(IV(zz+c)[mm],yv[mm]),bounds=(-0.3,0.3),method="bounded")
    return -r_.fun
base=sc(z,m)
P(f"기준(+0.08/+0.06 적용) {base:.2f}")
P("== 1) 13투 잔차 by 월 (보정 후, 남은 잔차)")
r0=minimize_scalar(lambda c:-bss(IV(z+c)[m],yv[m]),bounds=(-0.3,0.3),method="bounded")
pz=IV(z+r0.x)
for M in (3,4,5,6,7,8,9):
    mm=t13p&(mo==M)
    if mm.sum()<800: continue
    P(f"  {M}월 n={int(mm.sum())}  {100*(yv[mm].mean()-pz[mm].mean()):+.2f}pp")
P("== 2) 13투 cold/stale (보정 후 잔차)")
for tag,mm in (("cold13투",coldp&(pt==13)&~isf),("stale13투",st&(pt==13)),("warm13투",t13p&~coldp&~st)):
    if mm.sum()<200: P(f"  {tag} n={int(mm.sum())} 소표본"); continue
    P(f"  {tag} n={int(mm.sum())}  {100*(yv[mm].mean()-pz[mm].mean()):+.2f}pp")
P("== 3) 불일치행 연화 — iso 분위별 잔차/보정")
iso=np.abs(Z4-np.median(Z4,axis=1,keepdims=True)).max(1)
q=np.quantile(iso[m],[0.5,0.8,0.95])
P(f"  iso 분위 {q.round(3)}")
for tag,mm in (("합의(iso<q50)",m&(iso<q[0])),("중간",m&(iso>=q[0])&(iso<q[2])),("불일치(top5%)",m&(iso>=q[2]))):
    g=minimize_scalar(lambda c:-bss(IV(LG(pz[mm])+c),yv[mm]),bounds=(-0.5,0.5),method="bounded")
    P(f"  {tag} n={int(mm.sum())}  잔차 {100*(yv[mm].mean()-pz[mm].mean()):+.2f}pp  최적시프트 {g.x:+.4f}  이득 {-g.fun-bss(pz[mm],yv[mm]):+.1f}")
# 불일치행 수축(G2<1) 시험
for G2 in (0.9,0.8):
    z2=z.copy()
    hh=iso>=q[2]
    z0=np.average(z[m])
    z2=np.where(hh,z0+G2*(z-z0),z2)
    P(f"  연화 G2={G2} (top5%)  Δ {sc(z2,m)-base:+.2f}")
