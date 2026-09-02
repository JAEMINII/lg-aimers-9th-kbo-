# -*- coding: utf-8 -*-
"""성분분해 확장 스윕 — 투수측 ball/strike + 대안 키, 타자측 middle. 3폴드.
배치보정 3종 고정 가산 위에서 각 후보 β 스윕 (comp_prior.py 확장)."""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
def P(s): sys.stdout.buffer.write((s+"\n").encode("utf-8")); sys.stdout.flush()
DL=os.environ.get("CP_DL","colab/_dl"); DATA=os.environ.get("CP_DATA","open (1)/data")
W4=np.array([0.29,0.12,0.29,0.30]); CW=np.array([0.45,0.10,0.25,0.20])
def bss(p,t):
    r=t.mean(); return 100000*(1-((p-t)**2).mean()/(r*(1-r)))
def LG(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def IV(z): return 1/(1+np.exp(-z))
def ld(*n):
    a=[np.load(os.path.join(DL,x)).astype(np.float64) for x in n]
    a=[x.mean(0) if x.ndim==2 else x for x in a]; return np.mean(a,0)
use=["row_id","season","game_type","pitcher_id","batter_id","batter_hand","pitcher_hand",
     "balls_before","strikes_before","outs_before","inning","control_success",
     "asof_pitcher_n","asof_pitcher_reverse_rate","asof_pitcher_ball_rate",
     "asof_pitcher_strike_rate","asof_batter_n","asof_batter_middle_rate"]
tr=pd.read_csv(os.path.join(DATA,"train.csv"),encoding="utf-8-sig",usecols=use)
rid=tr["row_id"].str.slice(6).astype("int64")
tr=tr.assign(_r=rid).sort_values("_r").reset_index(drop=True)
y=tr["control_success"].to_numpy(float)
def recover(ent_col, n_col, rate_col):
    n=tr[n_col].fillna(0).to_numpy(float)
    cnt=tr[rate_col].fillna(0).to_numpy(float)*n
    eid=tr[ent_col].to_numpy()
    ordidx=np.arange(len(tr))
    nxt=pd.DataFrame({"e":eid,"i":ordidx}).groupby("e")["i"].shift(-1).to_numpy()
    ok=~np.isnan(nxt); nx=nxt[ok].astype(int); cur=ordidx[ok]
    good=np.abs(n[nx]-n[cur]-1)<1e-6
    out=np.full(len(tr),np.nan)
    out[cur[good]]=np.round(cnt[nx[good]]-cnt[cur[good]])
    out[~np.isin(out,[0,1])]=np.nan
    return out
p_ball=recover("pitcher_id","asof_pitcher_n","asof_pitcher_ball_rate")
p_str=recover("pitcher_id","asof_pitcher_n","asof_pitcher_strike_rate")
p_rev=recover("pitcher_id","asof_pitcher_n","asof_pitcher_reverse_rate")
b_mid=recover("batter_id","asof_batter_n","asof_batter_middle_rate")
P(f"복원율 p_ball {np.mean(~np.isnan(p_ball)):.3f}({np.nanmean(p_ball):.3f})  p_str {np.mean(~np.isnan(p_str)):.3f}({np.nanmean(p_str):.3f})  b_mid {np.mean(~np.isnan(b_mid)):.3f}({np.nanmean(b_mid):.3f})")
pid=tr["pitcher_id"].to_numpy().astype("int64")
bid=tr["batter_id"].to_numpy().astype("int64")
c12=(tr["balls_before"].to_numpy()*3+tr["strikes_before"].to_numpy()).astype("int64")
bh=tr["batter_hand"].to_numpy().astype("int64")
ph=tr["pitcher_hand"].to_numpy().astype("int64")
outs=tr["outs_before"].to_numpy().astype("int64")
inn=np.clip(tr["inning"].to_numpy(),1,9).astype("int64")
CANDS={
 "pball_hc": (p_ball, pid*1000+bh*100+c12, pid, 1000),
 "pstr_hc":  (p_str,  pid*1000+bh*100+c12, pid, 1000),
 "prev_out": (p_rev,  pid*10+outs,          pid, 10),
 "prev_inn": (p_rev,  pid*10+inn,           pid, 10),
 "prev_c12": (p_rev,  pid*100+c12,          pid, 100),
 "bmid_pc":  (b_mid,  bid*1000+ph*100+c12,  bid, 1000),
 "bmid_c12": (b_mid,  bid*100+c12,          bid, 100),
}
season=tr["season"].to_numpy(); isf=tr["game_type"].astype(str).to_numpy()=="F"
kp_c12=pid*100+c12; kp_out=pid*10+outs; kb_c12=bid*100+c12
CFG={2024:("ta2024_base.npy",None,"dg_2024_DIN.npy","msr_state6","msp_2024_ms_s2","msif_2024"),
     2022:("cb50fixed_2022.npy","h2h_2022_friend_s42.npy","dg_2022_DIN.npy","msr22_state6","msp_2022_ms_s2","msif_2022"),
     2023:("cb50_2023.npy","h2h_2023_friend_s42.npy","dg_2023_DIN.npy","msr23_state6","msr23_ms_s2","msif_2023")}
for VS in (2024,2022,2023):
    trm2=season<VS
    def sdev(vals, kk, ent, div, vm=None):
        mm=trm2&~np.isnan(vals) if vm is None else trm2&vm&~np.isnan(vals)
        v=vals[mm]; k_=kk[mm]; e_=ent[mm]
        mu_=v.mean()
        pgg=pd.Series(v).groupby(e_).agg(["sum","count"])
        prr=(pgg["sum"]+300*mu_)/(pgg["count"]+300)
        kgg=pd.Series(v).groupby(k_).agg(["sum","count"])
        kent=(pd.Series(k_)//div).groupby(k_).first()
        base_=prr.reindex(kent.values).to_numpy()
        krt=(kgg["sum"].to_numpy()+300*base_)/(kgg["count"].to_numpy()+300)
        return dict(zip(kgg.index.to_numpy(),krt-base_))
    cbn,tbn,dnn,m6p,mop,mifp=CFG[VS]
    va_m=season==VS
    va=tr[va_m]; trn=tr[trm2]
    yv=y[va_m]; isf_v=isf[va_m]
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
    Fm=np.column_stack([cb,tab,din,r]); Wm=np.tile(W4,(nn,1)); Wm[cold]=CW
    pmix=(Fm*Wm).sum(1)
    m=~isf_v if VS==2023 else np.ones(nn,bool)
    ysucc=y.copy()
    d1=sdev(ysucc,kp_c12,pid,100); d2=sdev(ysucc,kp_out,pid,10); d3=sdev(ysucc,kb_c12,bid,100)
    z0=LG(pmix)+0.12*np.array([d1.get(k,0.) for k in kp_c12[va_m]])\
       -0.10*np.array([d2.get(k,0.) for k in kp_out[va_m]])\
       +0.05*np.array([d3.get(k,0.) for k in kb_c12[va_m]])
    # rev 카드(배치 예정)도 고정 가산 -0.3
    drev=sdev(p_rev,pid*1000+bh*100+c12,pid,1000)
    z0=z0-0.30*np.array([drev.get(k,0.) for k in (pid*1000+bh*100+c12)[va_m]])
    def sc(z):
        r_=minimize_scalar(lambda c:-bss(IV(z+c)[m],yv[m]),bounds=(-0.3,0.3),method="bounded")
        return -r_.fun
    base=sc(z0)
    out=f"VS{VS} 기준 {base:.1f} | "
    for name,(vals,kk,ent,div) in CANDS.items():
        dd=sdev(vals,kk,ent,div)
        dvv=np.array([dd.get(k,0.0) for k in kk[va_m]])
        best=(0,0.0)
        for b in (-0.6,-0.45,-0.3,-0.15,0.15,0.3,0.45,0.6):
            d=sc(z0+b*dvv)-base
            if d>best[1]: best=(b,d)
        out+=f"{name} β{best[0]:+.2f} {best[1]:+.2f} | "
    P(out)
P("comp2 끝")
