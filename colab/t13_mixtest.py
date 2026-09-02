# -*- coding: utf-8 -*-
"""cbh9 수술 CB 를 믹스 CB 슬롯에 넣었을 때 (VS2024 폴드, 대조 cbh4_b58)."""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
def P(s): sys.stdout.buffer.write((s+"\n").encode("utf-8"))
DL="colab/_dl"; DATA="open (1)/data"
W4=np.array([0.29,0.12,0.29,0.30]); CW=np.array([0.45,0.10,0.25,0.20])
def bss(p,t):
    r=t.mean(); return 100000*(1-((p-t)**2).mean()/(r*(1-r)))
def LG(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def IV(z): return 1/(1+np.exp(-z))
def sc(p,t):
    r=minimize_scalar(lambda c:-bss(IV(LG(p)+c),t),bounds=(-0.3,0.3),method="bounded")
    return -r.fun
def ld(*n):
    a=[np.load(os.path.join(DL,x)).astype(np.float64) for x in n]
    a=[x.mean(0) if x.ndim==2 else x for x in a]; return np.mean(a,0)
tr=pd.read_csv(os.path.join(DATA,"train.csv"),encoding="utf-8-sig",
    usecols=["row_id","season","game_month","game_type","pitcher_id","batter_id","control_success"])
rid=tr["row_id"].str.slice(6).astype("int64")
tr=tr.assign(_r=rid).sort_values("_r").reset_index(drop=True)
va=tr[tr.season==2024]; trn=tr[tr.season<2024]
yv=va["control_success"].to_numpy(np.float64)
isf=va["game_type"].astype(str).to_numpy()=="F"
pid=va["pitcher_id"].to_numpy(); bid=va["batter_id"].to_numpy()
wp=set(trn["pitcher_id"]); wb=set(trn["batter_id"])
lp=trn.groupby("pitcher_id")["season"].max()
tab=(0.6*np.load(f"{DL}/c4g_2024_c4_all_s42.npy")+0.4*np.load(f"{DL}/c4g_2024_c4_regular_s42.npy")).astype(float)
din=ld("dg_2024_DIN.npy"); ms6=ld("msr_state6_s42.npy","msr_state6_s1.npy")
mso=ld("msp_2024_ms_s2_s42.npy","msp_2024_ms_s2_s1.npy"); mif=ld("msif_2024_s42.npy","msif_2024_s1.npy")
n=len(yv)
cold=np.array([p not in wp for p in pid])|np.array([b not in wb for b in bid])
lsp=np.array([lp.get(v,-1) for v in pid])
st=~cold&(lsp<=2022)&~isf; idf=(cold&~isf)|st
r=np.where(idf,mif,np.where(isf,mso,ms6))
def mixwith(cb):
    F=np.column_stack([cb,tab,din,r]); Wm=np.tile(W4,(n,1)); Wm[cold]=CW
    return (F*Wm).sum(1)
base=sc(mixwith(np.load(f"{DL}/cbh4_2024_b58.npy"))[~isf],yv[~isf])
P(f"기준(b58) {base:.2f}")
for arm in ("w0","w05","t13br"):
    f=f"{DL}/cbh9_2024_{arm}.npy"
    if not os.path.exists(f): continue
    d=sc(mixwith(np.load(f))[~isf],yv[~isf])-base
    P(f"믹스 {arm} {d:+.2f}")
