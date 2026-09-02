# -*- coding: utf-8 -*-
"""t13 내부 선수별 잔차 이질성의 연도 전이 — 2023폴드(5월+) vs 2024폴드."""
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
rid=tr["row_id"].str.slice(6).astype("int64")
tr=tr.assign(_r=rid).sort_values("_r").reset_index(drop=True)
y=tr["control_success"].to_numpy(float)
season=tr["season"].to_numpy()
CFG={2024:("ta2024_base.npy",None,"dg_2024_DIN.npy","msr_state6","msp_2024_ms_s2","msif_2024"),
     2023:("cb50_2023.npy","h2h_2023_friend_s42.npy","dg_2023_DIN.npy","msr23_state6","msr23_ms_s2","msif_2023")}
DEV={}
for VS in (2024,2023):
    cbn,tbn,dnn,m6p,mop,mifp=CFG[VS]
    va_m=season==VS
    va=tr[va_m]; trn=tr[season<VS]
    yv=y[va_m]
    isf=va["game_type"].astype(str).to_numpy()=="F"
    wp=set(trn["pitcher_id"]); wb=set(trn["batter_id"])
    lp=trn.groupby("pitcher_id")["season"].max()
    cb=ld(cbn)
    if VS==2024:
        tab=(0.6*np.load(f"{DL}/c4g_2024_c4_all_s42.npy")+0.4*np.load(f"{DL}/c4g_2024_c4_regular_s42.npy")).astype(float)
    else: tab=ld(tbn)
    din=ld(dnn); ms6=ld(m6p+"_s42.npy",m6p+"_s1.npy"); mso=ld(mop+"_s42.npy",mop+"_s1.npy")
    mif=ld(mifp+"_s42.npy",mifp+"_s1.npy")
    nn=len(yv)
    pidv=va["pitcher_id"].to_numpy(); bidv=va["batter_id"].to_numpy()
    cold=np.array([p not in wp for p in pidv])|np.array([b not in wb for b in bidv])
    lsp=np.array([lp.get(v,-1) for v in pidv])
    st=~cold&(lsp<=VS-2)&~isf; idf=(cold&~isf)|st
    r=np.where(idf,mif,np.where(isf,mso,ms6))
    F=np.column_stack([cb,tab,din,r]); Wm=np.tile(W4,(nn,1)); Wm[cold]=CW
    z=LG((F*Wm).sum(1))
    m=~isf
    r0=minimize_scalar(lambda c:-bss(IV(z+c)[m],yv[m]),bounds=(-0.3,0.3),method="bounded")
    pz=IV(z+r0.x)
    e=yv-pz
    pt=va.pitcher_team_id.to_numpy(); bt=va.batter_team_id.to_numpy()
    mo=va.game_month.to_numpy()
    post=(mo>=5) if VS==2023 else np.ones(nn,bool)
    for side,mask,ids in (("투",(pt==13)&m&post,pidv),("타",(bt==13)&(pt!=13)&m&post,bidv)):
        ee=e[mask]; ii=ids[mask]
        mu=ee.mean()
        g=pd.DataFrame({"i":ii,"e":ee-mu}).groupby("i")["e"].agg(["sum","count"])
        g=g[g["count"]>=150]
        DEV.setdefault(side,{})[VS]=dict(dev=(g["sum"]/(g["count"]+300)).to_dict(),
                                          n=g["count"].to_dict())
for side in ("투","타"):
    a=DEV[side][2023]; b=DEV[side][2024]
    common=sorted(set(a["dev"])&set(b["dev"]))
    if len(common)<5: P(f"{side}측 공통 선수 {len(common)} — 판정불가"); continue
    x=np.array([a["dev"][k] for k in common]); yy=np.array([b["dev"][k] for k in common])
    w=np.array([min(a["n"][k],b["n"][k]) for k in common],float)
    xm=np.average(x,weights=w); ym=np.average(yy,weights=w)
    cov=np.average((x-xm)*(yy-ym),weights=w)
    corr=cov/np.sqrt(np.average((x-xm)**2,weights=w)*np.average((yy-ym)**2,weights=w))
    P(f"13{side}측  공통 {len(common)}명  가중상관(23->24) {corr:+.3f}  "
      f"23편차SD {100*x.std():.2f}pp  24편차SD {100*yy.std():.2f}pp")
