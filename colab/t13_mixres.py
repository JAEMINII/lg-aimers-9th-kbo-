# -*- coding: utf-8 -*-
"""t13 믹스 잔차 — VS=2024/2023 폴드에서 t13 조각의 체계적 편향 측정."""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
def P(s): sys.stdout.buffer.write((s+"\n").encode("utf-8"))
SC="colab"; DL=os.path.join(SC,"_dl"); DATA="open (1)/data"
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
    usecols=["row_id","season","game_month","game_type","pitcher_id","batter_id",
             "pitcher_team_id","batter_team_id","control_success"])
rid=tr["row_id"].astype(str).str.extract(r"(\d+)$",expand=False).astype("int64")
tr=tr.assign(_rid=rid).sort_values("_rid",kind="mergesort").reset_index(drop=True)
CFG={2024:("ta2024_base.npy",None,"dg_2024_DIN.npy","msr_state6","msp_2024_ms_s2","msif_2024"),
     2023:("cb50_2023.npy","h2h_2023_friend_s42.npy","dg_2023_DIN.npy","msr23_state6","msr23_ms_s2","msif_2023")}
for VS in (2024,2023):
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
    st=~cold&(lsp<=VS-2)&~isf
    idf=(cold&~isf)|st
    r=np.where(idf,mif,np.where(isf,mso,ms6))
    F=np.column_stack([cb,tab,din,r]); Wm=np.tile(W4,(n,1)); Wm[cold]=CW
    p=(F*Wm).sum(1)
    m = ~isf
    t13=((va.pitcher_team_id==13)|(va.batter_team_id==13)).to_numpy()
    half=np.where(va.game_month.to_numpy()<=6,"H1","H2")
    # 전역 시프트 먼저 (수준보정 분리)
    r0=minimize_scalar(lambda c:-bss(IV(LG(p[m])+c),yv[m]),bounds=(-0.3,0.3),method="bounded")
    c0=r0.x; pz=IV(LG(p)+c0)
    P(f"== VS{VS} (R행, 전역시프트 {c0:+.4f} 적용후)  기준 {bss(pz[m],yv[m]):.1f}")
    for tag,mask in (("t13",t13&m),("비13",~t13&m)):
        for hf in ("H1","H2"):
            mm=mask&(half==hf)
            if mm.sum()==0: continue
            b=yv[mm].mean()-pz[mm].mean()
            g=minimize_scalar(lambda c:-bss(IV(LG(pz[mm])+c),yv[mm]),bounds=(-0.5,0.5),method="bounded")
            P(f"  {tag} {hf}  n={int(mm.sum())}  잔차 {b*100:+.2f}pp  최적로짓 {g.x:+.4f}  조각이득 {-g.fun-bss(pz[mm],yv[mm]):+.1f}")
    # 롤링: H1 에서 t13 오프셋 적합 -> H2 t13 에 적용, 전체 R 점수 변화
    mmf=t13&m&(half=="H1"); mma=t13&m&(half=="H2")
    g1=minimize_scalar(lambda c:-bss(IV(LG(pz[mmf])+c),yv[mmf]),bounds=(-0.5,0.5),method="bounded").x
    z=LG(pz.copy()); z[mma]+=g1
    pa=IV(z)
    P(f"  롤링(H1적합 {g1:+.4f} ->H2적용) 전체R 변화 {bss(pa[m&(half=='H2')],yv[m&(half=='H2')])-bss(pz[m&(half=='H2')],yv[m&(half=='H2')]):+.2f}")
