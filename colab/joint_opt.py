# -*- coding: utf-8 -*-
"""83 전역계수 공동 재최적화 — 가중4+β5+샤픈G, 83 앵커 정규화, 3폴드 목적."""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import minimize, minimize_scalar
def P(s): sys.stdout.buffer.write((s+"\n").encode("utf-8")); sys.stdout.flush()
DL="colab/_dl"; DATA="open (1)/data"
def bss(p,t):
    r=t.mean(); return 100000*(1-((p-t)**2).mean()/(r*(1-r)))
def LG(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def IV(z): return 1/(1+np.exp(-z))
def ld(*n):
    a=[np.load(os.path.join(DL,x)).astype(np.float64) for x in n]
    a=[x.mean(0) if x.ndim==2 else x for x in a]; return np.mean(a,0)
use=["row_id","season","game_month","game_type","pitcher_id","batter_id","batter_hand",
     "balls_before","strikes_before","outs_before","pitcher_team_id","batter_team_id",
     "control_success","asof_pitcher_n","asof_pitcher_reverse_rate","asof_pitcher_strike_rate"]
tr=pd.read_csv(os.path.join(DATA,"train.csv"),encoding="utf-8-sig",usecols=use)
rid=tr["row_id"].str.slice(6).astype("int64")
tr=tr.assign(_r=rid).sort_values("_r").reset_index(drop=True)
y=tr["control_success"].to_numpy(float)
pid=tr["pitcher_id"].to_numpy().astype("int64"); bid=tr["batter_id"].to_numpy().astype("int64")
c12=(tr["balls_before"].to_numpy()*3+tr["strikes_before"].to_numpy()).astype("int64")
bh=tr["batter_hand"].to_numpy().astype("int64"); outs=tr["outs_before"].to_numpy().astype("int64")
season=tr["season"].to_numpy(); isf_all=tr["game_type"].astype(str).to_numpy()=="F"
def recover(rcol):
    n=tr["asof_pitcher_n"].fillna(0).to_numpy(float); cnt=tr[rcol].fillna(0).to_numpy(float)*n
    oi=np.arange(len(tr))
    nxt=pd.DataFrame({"e":pid,"i":oi}).groupby("e")["i"].shift(-1).to_numpy()
    ok=~np.isnan(nxt); nx=nxt[ok].astype(int); cur=oi[ok]
    g=np.abs(n[nx]-n[cur]-1)<1e-6
    out=np.full(len(tr),np.nan); out[cur[g]]=np.round(cnt[nx[g]]-cnt[cur[g]])
    out[~np.isin(out,[0,1])]=np.nan; return out
p_rev=recover("asof_pitcher_reverse_rate"); p_str=recover("asof_pitcher_strike_rate")
kp_c12=pid*100+c12; kp_out=pid*10+outs; kb_c12=bid*100+c12; k_hc=pid*1000+bh*100+c12
CFG={2024:("ta2024_base.npy",None,"dg_2024_DIN.npy","msr_state6","msp_2024_ms_s2","msif_2024"),
     2022:("cb50fixed_2022.npy","h2h_2022_friend_s42.npy","dg_2022_DIN.npy","msr22_state6","msp_2022_ms_s2","msif_2022"),
     2023:("cb50_2023.npy","h2h_2023_friend_s42.npy","dg_2023_DIN.npy","msr23_state6","msr23_ms_s2","msif_2023")}
FD={}
for VS in (2024,2022,2023):
    trm2=season<VS
    def sdev(vals,kk,ent,div):
        mm=trm2&~np.isnan(vals)
        v=vals[mm];k_=kk[mm];e_=ent[mm];mu_=v.mean()
        pg=pd.Series(v).groupby(e_).agg(["sum","count"]);pr=(pg["sum"]+300*mu_)/(pg["count"]+300)
        kg=pd.Series(v).groupby(k_).agg(["sum","count"])
        ke=(pd.Series(k_)//div).groupby(k_).first()
        b_=pr.reindex(ke.values).to_numpy()
        return dict(zip(kg.index.to_numpy(),(kg["sum"].to_numpy()+300*b_)/(kg["count"].to_numpy()+300)-b_))
    cbn,tbn,dnn,m6p,mop,mifp=CFG[VS]
    va_m=season==VS
    va=tr[va_m]; trn=tr[trm2]
    yv=y[va_m]; isf_v=isf_all[va_m]
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
    st=~cold&(lsp<=VS-2)&~isf_v; idf=(cold&~isf_v)|st
    r=np.where(idf,mif,np.where(isf_v,mso,ms6))
    Z4=np.column_stack([LG(cb),LG(tab),LG(din),LG(r)])
    d1=sdev(y,kp_c12,pid,100);d2=sdev(y,kp_out,pid,10);d3=sdev(y,kb_c12,bid,100)
    dr=sdev(p_rev,k_hc,pid,1000);ds=sdev(p_str,k_hc,pid,1000)
    D=np.column_stack([
        np.array([d1.get(k,0.) for k in kp_c12[va_m]]),
        np.array([d2.get(k,0.) for k in kp_out[va_m]]),
        np.array([d3.get(k,0.) for k in kb_c12[va_m]]),
        np.array([dr.get(k,0.) for k in k_hc[va_m]]),
        np.array([ds.get(k,0.) for k in k_hc[va_m]])])
    pt=va.pitcher_team_id.to_numpy(); bt=va.batter_team_id.to_numpy()
    t13c=np.where((pt==13)&~isf_v,0.08,0)+np.where((bt==13)&(pt!=13)&~isf_v,0.06,0)
    m=~isf_v if VS==2023 else np.ones(nn,bool)
    FD[VS]=dict(Z4=Z4,D=D,t13=t13c,yv=yv,cold=cold,m=m)
W83=np.array([0.3928,0.0600,0.1665,0.3808]); CW=np.array([0.45,0.10,0.25,0.20])
B83=np.array([0.12,-0.10,0.05,-0.30,-0.45]); G83=1.05; TH=0.08
def score(theta, VS):
    F=FD[VS]
    w=theta[:4]; betas=theta[4:9]; G=theta[9]
    w=np.clip(w,0.01,0.7); w=w/w.sum()
    P4=IV(F["Z4"])
    pm=P4@w
    pm=np.where(F["cold"],P4@CW,pm)
    z=LG(pm)+F["D"]@betas+F["t13"]
    iso=np.abs(F["Z4"]-np.median(F["Z4"],axis=1,keepdims=True)).max(1)
    z0=np.average(z)
    z=np.where(iso<TH,z0+G*(z-z0),z)
    m=F["m"]
    r_=minimize_scalar(lambda c:-bss(IV(z+c)[m],F["yv"][m]),bounds=(-0.3,0.3),method="bounded")
    return -r_.fun
th0=np.concatenate([W83,B83,[G83]])
s0={VS:score(th0,VS) for VS in (2024,2023,2022)}
P(f"83 기준: 2024 {s0[2024]:.2f}  2023 {s0[2023]:.2f}  2022 {s0[2022]:.2f}")
RHO=2000.0
def obj(th):
    return -(score(th,2024)+0.5*score(th,2023)+0.3*score(th,2022)) \
           + RHO*np.sum((th-th0)**2)
res=minimize(obj,th0,method="Nelder-Mead",
             options=dict(maxfev=400,xatol=1e-3,fatol=0.05))
th=res.x
P(f"이동: w {np.round(th[:4]/th[:4].sum(),4)}  β {np.round(th[4:9],3)}  G {th[9]:.3f}")
for VS in (2024,2023,2022):
    P(f"  VS{VS}: {score(th,VS)-s0[VS]:+.2f}")
np.save("colab/_dl/joint_theta.npy", th)
