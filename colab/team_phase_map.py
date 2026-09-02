# -*- coding: utf-8 -*-
"""팀x역할x위상 잔차 맵 — 전년맵->이듬해 같은위상 전이 검증 (조언 라운드).
맵: 비13 R행, 위상(early34/mid57/late810), 투/타 백피팅 공동추정 + 수축 + 중심화.
적용: z += B * 4 * (a[pt,ph] + b[bt,ph]), 비13 R행만. 채점: 위상 3~7월 R행."""
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
season=tr["season"].to_numpy(); isf=tr["game_type"].astype(str).to_numpy()=="F"
mo=tr["game_month"].to_numpy()
phase=np.where(mo<=4,0,np.where(mo<=7,1,2))
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
def fold_z(VS):
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
    z=LG((Fm*Wm).sum(1))
    d1=sdev(y,kp_c12,pid,100);d2=sdev(y,kp_out,pid,10);d3=sdev(y,kb_c12,bid,100)
    dr=sdev(p_rev,k_hc,pid,1000);ds=sdev(p_str,k_hc,pid,1000)
    z=z+0.12*np.array([d1.get(k,0.) for k in kp_c12[va_m]])\
       -0.10*np.array([d2.get(k,0.) for k in kp_out[va_m]])\
       +0.05*np.array([d3.get(k,0.) for k in kb_c12[va_m]])\
       -0.30*np.array([dr.get(k,0.) for k in k_hc[va_m]])\
       -0.45*np.array([ds.get(k,0.) for k in k_hc[va_m]])
    pt=va.pitcher_team_id.to_numpy(); bt=va.batter_team_id.to_numpy()
    z=z+np.where((pt==13)&~isf_v,0.08,0)+np.where((bt==13)&(pt!=13)&~isf_v,0.06,0)
    m=~isf_v
    r0=minimize_scalar(lambda c:-bss(IV(z+c)[m],yv[m]),bounds=(-0.3,0.3),method="bounded")
    return dict(z=z+r0.x, yv=yv, isf=isf_v, pt=pt, bt=bt,
                mo=va.game_month.to_numpy(), ph=phase[va_m])
def build_map(F, KAP):
    """비13 R행 잔차 -> a[pt,ph], b[bt,ph] 백피팅 수축 + 위상중심화. 확률단위."""
    m=~F["isf"]&(F["pt"]!=13)&(F["bt"]!=13)
    e=(F["yv"]-IV(F["z"]))[m]
    pt=F["pt"][m]; bt=F["bt"][m]; ph=F["ph"][m]
    teams=sorted(set(pt)|set(bt))
    a={(t,p):0.0 for t in teams for p in range(3)}
    b={(t,p):0.0 for t in teams for p in range(3)}
    for _ in range(4):
        res=e-np.array([b[(t,p)] for t,p in zip(bt,ph)])
        df=pd.DataFrame({"t":pt,"p":ph,"r":res}).groupby(["t","p"])["r"].agg(["sum","count"])
        for (t,p),row in df.iterrows(): a[(t,p)]=row["sum"]/(row["count"]+KAP)
        for p in range(3):
            mu=np.mean([a[(t,p)] for t in teams]); 
            for t in teams: a[(t,p)]-=mu
        res=e-np.array([a[(t,p)] for t,p in zip(pt,ph)])
        df=pd.DataFrame({"t":bt,"p":ph,"r":res}).groupby(["t","p"])["r"].agg(["sum","count"])
        for (t,p),row in df.iterrows(): b[(t,p)]=row["sum"]/(row["count"]+KAP)
        for p in range(3):
            mu=np.mean([b[(t,p)] for t in teams])
            for t in teams: b[(t,p)]-=mu
    return a,b
def apply_map(F,a,b,B):
    corr=np.array([a.get((t,p),0.)+ (0 if t2 is None else 0) for t,p,t2 in zip(F["pt"],F["ph"],F["pt"])])
    corr=np.array([a.get((t,p),0.) for t,p in zip(F["pt"],F["ph"])])+\
         np.array([b.get((t,p),0.) for t,p in zip(F["bt"],F["ph"])])
    mask=(~F["isf"])&(F["pt"]!=13)&(F["bt"]!=13)
    return F["z"]+np.where(mask,B*4.0*corr,0.0)
Z={VS:fold_z(VS) for VS in (2022,2023,2024)}
for KAP in (3000,10000):
    # 전이 1: 2022 잔차맵 -> 2023 (R만)
    a,b=build_map(Z[2022],KAP)
    F=Z[2023]; m=(~F["isf"])&(F["mo"]>=3)&(F["mo"]<=7)
    base=bss(IV(F["z"])[m],F["yv"][m])
    o1=" ".join(f"B{B}: {bss(IV(apply_map(F,a,b,B))[m],F['yv'][m])-base:+.2f}" for B in (0.5,1.0))
    # 전이 2: 2023 잔차맵 -> 2024
    a2,b2=build_map(Z[2023],KAP)
    F=Z[2024]; m=(~F["isf"])&(F["mo"]>=3)&(F["mo"]<=7)
    base=bss(IV(F["z"])[m],F["yv"][m])
    o2=" ".join(f"B{B}: {bss(IV(apply_map(F,a2,b2,B))[m],F['yv'][m])-base:+.2f}" for B in (0.5,1.0))
    P(f"κ={KAP}  22맵->23: {o1}   |   23맵->24: {o2}")
# 위생검사: 배치용 2024 잔차맵의 t13 셀 (적용 안 함, 확인만)
a3,b3=build_map(Z[2024],3000)
P("2024맵 (배치용, κ3000) 팀별 early/mid [투수측 a]:")
for t in sorted(set(Z[2024]["pt"])):
    P(f"  팀{int(t):2d}  early {100*a3.get((t,0),0):+.2f}pp  mid {100*a3.get((t,1),0):+.2f}pp  late {100*a3.get((t,2),0):+.2f}pp")
