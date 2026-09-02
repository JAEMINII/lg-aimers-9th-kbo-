# -*- coding: utf-8 -*-
"""t13 잔차의 투수측/타자측 분해 + 월별 궤적 (VS=2024 폴드)."""
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
             "pitcher_team_id","batter_team_id","control_success"])
rid=tr["row_id"].astype(str).str.extract(r"(\d+)$",expand=False).astype("int64")
tr=tr.assign(_rid=rid).sort_values("_rid",kind="mergesort").reset_index(drop=True)
va=tr[tr.season==2024]; trn=tr[tr.season<2024]
yv=va["control_success"].to_numpy(np.float64)
isf=va["game_type"].astype(str).to_numpy()=="F"
pid=va["pitcher_id"].to_numpy(); bid=va["batter_id"].to_numpy()
wp=set(trn["pitcher_id"]); wb=set(trn["batter_id"])
lp=trn.groupby("pitcher_id")["season"].max()
cb=ld("ta2024_base.npy")
tab=(0.6*np.load(f"{DL}/c4g_2024_c4_all_s42.npy")+0.4*np.load(f"{DL}/c4g_2024_c4_regular_s42.npy")).astype(float)
din=ld("dg_2024_DIN.npy"); ms6=ld("msr_state6_s42.npy","msr_state6_s1.npy")
mso=ld("msp_2024_ms_s2_s42.npy","msp_2024_ms_s2_s1.npy"); mif=ld("msif_2024_s42.npy","msif_2024_s1.npy")
n=len(yv)
cold=np.array([p not in wp for p in pid])|np.array([b not in wb for b in bid])
lsp=np.array([lp.get(v,-1) for v in pid])
st=~cold&(lsp<=2022)&~isf; idf=(cold&~isf)|st
r=np.where(idf,mif,np.where(isf,mso,ms6))
F=np.column_stack([cb,tab,din,r]); Wm=np.tile(W4,(n,1)); Wm[cold]=CW
p=(F*Wm).sum(1)
m=~isf
r0=minimize_scalar(lambda c:-bss(IV(LG(p[m])+c),yv[m]),bounds=(-0.3,0.3),method="bounded")
pz=IV(LG(p)+r0.x)
pt=va.pitcher_team_id.to_numpy(); bt=va.batter_team_id.to_numpy()
mo=va.game_month.to_numpy()
p13=(pt==13)&~(bt==13); b13=(bt==13)&~(pt==13)
P("== 2024 폴드 R행 월별 잔차(pp): 13투수측만 / 13타자측만")
for M in sorted(set(mo[m])):
    for tag,mask in (("13투",p13),("13타",b13)):
        mm=mask&m&(mo==M)
        if mm.sum()<800: continue
        P(f"  2024-{int(M):02d} {tag}  n={int(mm.sum())}  {100*(yv[mm].mean()-pz[mm].mean()):+.2f}")
P("== 반기 합산")
half=np.where(mo<=6,"H1","H2")
for tag,mask in (("13투",p13),("13타",b13)):
    for hf in ("H1","H2"):
        mm=mask&m&(half==hf)
        b=100*(yv[mm].mean()-pz[mm].mean())
        g=minimize_scalar(lambda c:-bss(IV(LG(pz[mm])+c),yv[mm]),bounds=(-0.5,0.5),method="bounded")
        P(f"  {tag} {hf}  n={int(mm.sum())}  잔차 {b:+.2f}pp  최적로짓 {g.x:+.4f}  조각이득 {-g.fun-bss(pz[mm],yv[mm]):+.1f}")
