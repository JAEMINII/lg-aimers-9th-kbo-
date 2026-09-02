# -*- coding: utf-8 -*-
"""최근폼4 제거 CB 를 믹스에 넣은 판 vs b58 — 전체R/위상창/13투 조각."""
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
season=tr["season"].to_numpy(); isf_a=tr["game_type"].astype(str).to_numpy()=="F"
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
CFG={2024:(None,"dg_2024_DIN.npy","msr_state6","msp_2024_ms_s2","msif_2024"),
     2023:("h2h_2023_friend_s42.npy","dg_2023_DIN.npy","msr23_state6","msr23_ms_s2","msif_2023")}
for VS in (2024,):
    trm2=season<VS
    def sdev(vals,kk,ent,div):
        mm=trm2&~np.isnan(vals)
        v=vals[mm];k_=kk[mm];e_=ent[mm];mu_=v.mean()
        pg=pd.Series(v).groupby(e_).agg(["sum","count"]);pr=(pg["sum"]+300*mu_)/(pg["count"]+300)
        kg=pd.Series(v).groupby(k_).agg(["sum","count"])
        ke=(pd.Series(k_)//div).groupby(k_).first()
        b_=pr.reindex(ke.values).to_numpy()
        return dict(zip(kg.index.to_numpy(),(kg["sum"].to_numpy()+300*b_)/(kg["count"].to_numpy()+300)-b_))
    va_m=season==VS
    va=tr[va_m]; trn=tr[trm2]
    yv=y[va_m]; isf_v=isf_a[va_m]
    wp=set(trn["pitcher_id"]); wb=set(trn["batter_id"])
    lp=trn.groupby("pitcher_id")["season"].max()
    if VS==2024:
        tab=(0.6*np.load(f"{DL}/c4g_2024_c4_all_s42.npy")+0.4*np.load(f"{DL}/c4g_2024_c4_regular_s42.npy")).astype(float)
        din=ld("dg_2024_DIN.npy"); ms6=ld("msr_state6_s42.npy","msr_state6_s1.npy")
        mso=ld("msp_2024_ms_s2_s42.npy","msp_2024_ms_s2_s1.npy"); mif=ld("msif_2024_s42.npy","msif_2024_s1.npy")
    else:
        tab=ld("h2h_2023_friend_s42.npy"); din=ld("dg_2023_DIN.npy")
        ms6=ld("msr23_state6_s42.npy","msr23_state6_s1.npy")
        mso=ld("msr23_ms_s2_s42.npy","msr23_ms_s2_s1.npy"); mif=ld("msif_2023_s42.npy","msif_2023_s1.npy")
    nn=len(yv)
    pidv=va["pitcher_id"].to_numpy(); bidv=va["batter_id"].to_numpy()
    cold=np.array([p not in wp for p in pidv])|np.array([b not in wb for b in bidv])
    lsp=np.array([lp.get(v,-1) for v in pidv])
    st=~cold&(lsp<=VS-2)&~isf_v; idf=(cold&~isf_v)|st
    r=np.where(idf,mif,np.where(isf_v,mso,ms6))
    d1=sdev(y,kp_c12,pid,100);d2=sdev(y,kp_out,pid,10);d3=sdev(y,kb_c12,bid,100)
    dr=sdev(p_rev,k_hc,pid,1000);ds=sdev(p_str,k_hc,pid,1000)
    corr=0.12*np.array([d1.get(k,0.) for k in kp_c12[va_m]])\
        -0.10*np.array([d2.get(k,0.) for k in kp_out[va_m]])\
        +0.05*np.array([d3.get(k,0.) for k in kb_c12[va_m]])\
        -0.30*np.array([dr.get(k,0.) for k in k_hc[va_m]])\
        -0.45*np.array([ds.get(k,0.) for k in k_hc[va_m]])
    pt=va.pitcher_team_id.to_numpy(); bt=va.batter_team_id.to_numpy()
    t13c=np.where((pt==13)&~isf_v,0.08,0)+np.where((bt==13)&(pt!=13)&~isf_v,0.06,0)
    mo_v=va.game_month.to_numpy()
    mR=~isf_v; mPH=mR&(mo_v>=3)&(mo_v<=7); m13=(pt==13)&mR
    def zmix(cb):
        F=np.column_stack([cb,tab,din,r]); Wm=np.tile(W4,(nn,1)); Wm[cold]=CW
        return LG((F*Wm).sum(1))+corr+t13c
    out=f"VS{VS}  "
    zb=zmix(np.load(f"{DL}/cbh4_{VS}_b58.npy").astype(float))
    zn=zmix(np.load(f"{DL}/cbh13_bc32_l2300.npy").astype(float))
    for tag,mm in (("전체R",mR),("위상3-7",mPH),("13투",m13)):
        def sc(z):
            r_=minimize_scalar(lambda c:-bss(IV(z+c)[mm],yv[mm]),bounds=(-0.3,0.3),method="bounded")
            return -r_.fun
        out+=f"{tag} {sc(zn)-sc(zb):+.2f}  "
    # 13투 잔차 비교 (보정 전 관점)
    for tag,z in (("b58",zb),("nrf",zn)):
        pz=IV(z-t13c)  # t13 보정 빼고 잔차 측정
        r_=minimize_scalar(lambda c:-bss(IV(z[mR]-t13c[mR]+c),yv[mR]),bounds=(-0.3,0.3),method="bounded")
        pz=IV(z-t13c+r_.x)
        mm=m13&(mo_v>=3)&(mo_v<=7)
        out+=f"| {tag} 13투잔차 {100*(yv[mm].mean()-pz[mm].mean()):+.2f}pp "
    P(out)
