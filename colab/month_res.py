# -*- coding: utf-8 -*-
"""월별 수준 잔차 — 2024/2023/2022 폴드 R행, 전역시프트 후 월 오프셋의 일관성."""
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
    usecols=["row_id","season","game_month","game_type","pitcher_id","batter_id","control_success"])
rid=tr["row_id"].str.slice(6).astype("int64")
tr=tr.assign(_r=rid).sort_values("_r").reset_index(drop=True)
CFG={2024:("ta2024_base.npy",None,"dg_2024_DIN.npy","msr_state6","msp_2024_ms_s2","msif_2024"),
     2022:("cb50fixed_2022.npy","h2h_2022_friend_s42.npy","dg_2022_DIN.npy","msr22_state6","msp_2022_ms_s2","msif_2022"),
     2023:("cb50_2023.npy","h2h_2023_friend_s42.npy","dg_2023_DIN.npy","msr23_state6","msr23_ms_s2","msif_2023")}
for VS in (2024,2022,2023):
    cbn,tbn,dnn,m6p,mop,mifp=CFG[VS]
    va=tr[tr.season==VS]; trn=tr[tr.season<VS]
    yv=va["control_success"].to_numpy(np.float64)
    isf=va["game_type"].astype(str).to_numpy()=="F"
    pid=va["pitcher_id"].to_numpy(); bid=va["batter_id"].to_numpy()
    wp=set(trn["pitcher_id"]); wb=set(trn["batter_id"])
    lp=trn.groupby("pitcher_id")["season"].max()
    cb=ld(cbn)
    if VS==2024:
        tab=(0.6*np.load(f"{DL}/c4g_2024_c4_all_s42.npy")+0.4*np.load(f"{DL}/c4g_2024_c4_regular_s42.npy")).astype(float)
    else: tab=ld(tbn)
    din=ld(dnn); ms6=ld(m6p+"_s42.npy",m6p+"_s1.npy"); mso=ld(mop+"_s42.npy",mop+"_s1.npy")
    mif=ld(mifp+"_s42.npy",mifp+"_s1.npy")
    n=len(yv)
    cold=np.array([p not in wp for p in pid])|np.array([b not in wb for b in bid])
    lsp=np.array([lp.get(v,-1) for v in pid])
    st=~cold&(lsp<=VS-2)&~isf; idf=(cold&~isf)|st
    r=np.where(idf,mif,np.where(isf,mso,ms6))
    F=np.column_stack([cb,tab,din,r]); Wm=np.tile(W4,(n,1)); Wm[cold]=CW
    p=(F*Wm).sum(1)
    m=~isf
    r0=minimize_scalar(lambda c:-bss(IV(LG(p[m])+c),yv[m]),bounds=(-0.3,0.3),method="bounded")
    pz=IV(LG(p)+r0.x)
    mo=va.game_month.to_numpy()
    out=f"VS{VS} R행 월별 잔차pp: "
    for M in sorted(set(mo[m])):
        mm=m&(mo==M)
        if mm.sum()<3000: continue
        out+=f"{int(M)}월 {100*(yv[mm].mean()-pz[mm].mean()):+.2f}({int(mm.sum())//1000}k)  "
    P(out)
    # F행도
    mf=isf
    if mf.sum()>5000:
        outf=f"      F행: "
        for M in sorted(set(mo[mf])):
            mm=mf&(mo==M)
            if mm.sum()<2000: continue
            outf+=f"{int(M)}월 {100*(yv[mm].mean()-pz[mm].mean()):+.2f}  "
        P(outf)
