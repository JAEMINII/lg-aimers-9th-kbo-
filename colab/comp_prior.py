# -*- coding: utf-8 -*-
"""성분-prior (v85 아이디어 재구현) — as-of 차분으로 행별 reverse/middle 복원,
(투수,타자손,카운트12) 성분편차 lookup 을 믹스 로짓에 가산, 3폴드 β 스윕."""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
def P(s): sys.stdout.buffer.write((s+"\n").encode("utf-8"))
DL=os.environ.get("CP_DL","colab/_dl"); DATA=os.environ.get("CP_DATA","open (1)/data")
W4=np.array([0.29,0.12,0.29,0.30]); CW=np.array([0.45,0.10,0.25,0.20])
def bss(p,t):
    r=t.mean(); return 100000*(1-((p-t)**2).mean()/(r*(1-r)))
def LG(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def IV(z): return 1/(1+np.exp(-z))
def ld(*n):
    a=[np.load(os.path.join(DL,x)).astype(np.float64) for x in n]
    a=[x.mean(0) if x.ndim==2 else x for x in a]; return np.mean(a,0)
use=["row_id","season","game_type","pitcher_id","batter_id","batter_hand",
     "balls_before","strikes_before","outs_before","control_success","asof_pitcher_n",
     "asof_pitcher_reverse_rate","asof_pitcher_middle_rate"]
tr=pd.read_csv(os.path.join(DATA,"train.csv"),encoding="utf-8-sig",usecols=use)
rid=tr["row_id"].str.slice(6).astype("int64")
tr=tr.assign(_r=rid).sort_values("_r").reset_index(drop=True)
n=tr["asof_pitcher_n"].fillna(0).to_numpy(float)
crev=tr["asof_pitcher_reverse_rate"].fillna(0).to_numpy(float)*n
cmid=tr["asof_pitcher_middle_rate"].fillna(0).to_numpy(float)*n
pid=tr["pitcher_id"].to_numpy()
# 같은 투수의 다음 등장에서 이번 행 결과를 복원
ordidx=np.arange(len(tr))
df=pd.DataFrame({"pid":pid,"i":ordidx})
g=df.groupby("pid")["i"]
nxt=g.shift(-1).to_numpy()
ok=~np.isnan(nxt)
nx=nxt[ok].astype(int); cur=ordidx[ok]
dn=n[nx]-n[cur]
rev=np.full(len(tr),np.nan); mid=np.full(len(tr),np.nan)
good=np.abs(dn-1)<1e-6
rev[cur[good]]=np.round(crev[nx[good]]-crev[cur[good]])
mid[cur[good]]=np.round(cmid[nx[good]]-cmid[cur[good]])
val=(~np.isnan(rev))&(np.isin(rev,[0,1]))&(np.isin(mid,[0,1]))
y=tr["control_success"].to_numpy(float)
way=np.where(val&(y==0)&(rev==0)&(mid==0),1.0,np.nan)
way[val&((y==1)|(rev==1)|(mid==1))]=0.0
P(f"복원 커버 {val.mean():.3f}  reverse율 {np.nanmean(rev[val]):.4f}  middle율 {np.nanmean(mid[val]):.4f}  wayoff율 {np.nanmean(way[val]):.4f}")
c12=(tr["balls_before"].to_numpy()*3+tr["strikes_before"].to_numpy()).astype(int)
bh=tr["batter_hand"].to_numpy()
key=pid.astype("int64")*1000+bh.astype("int64")*100+c12
COMP={"rev":rev,"mid":mid,"way":way}
kp_c12=pid.astype("int64")*100+c12          # pcnt: 투수x카운트
kp_out=pid.astype("int64")*10+tr["outs_before"].to_numpy().astype("int64")
kb_c12=tr["batter_id"].to_numpy().astype("int64")*100+c12
season=tr["season"].to_numpy(); isf=tr["game_type"].astype(str).to_numpy()=="F"
CFG={2024:("ta2024_base.npy",None,"dg_2024_DIN.npy","msr_state6","msp_2024_ms_s2","msif_2024"),
     2022:("cb50fixed_2022.npy","h2h_2022_friend_s42.npy","dg_2022_DIN.npy","msr22_state6","msp_2022_ms_s2","msif_2022"),
     2023:("cb50_2023.npy","h2h_2023_friend_s42.npy","dg_2023_DIN.npy","msr23_state6","msr23_ms_s2","msif_2023")}
for VS in (2024,2022,2023):
    trm=(season<VS)&val
    va_m=season==VS
    # 계층수축: 키(κ300) -> 투수(κ300) -> 리그
    dev={}
    for cn,arr in COMP.items():
        v=arr[trm]; k=key[trm]; p_=pid[trm]
        lg_mu=np.nanmean(v)
        s=pd.Series(v)
        pg=s.groupby(p_).agg(["sum","count"])
        p_rate=(pg["sum"]+300*lg_mu)/(pg["count"]+300)
        kg=s.groupby(k).agg(["sum","count"])
        kp=pd.Series(k).groupby(k).first()  # key->pid 복원용
        kpid=(pd.Series(k)//1000).groupby(k).first()
        pr=p_rate.reindex(kpid.values).to_numpy()
        k_rate=(kg["sum"].to_numpy()+300*pr)/(kg["count"].to_numpy()+300)
        dev[cn]=dict(zip(kg.index.to_numpy(),k_rate-pr))
    cbn,tbn,dnn,m6p,mop,mifp=CFG[VS]
    va=tr[va_m]; trn=tr[season<VS]
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
    F=np.column_stack([cb,tab,din,r]); Wm=np.tile(W4,(nn,1)); Wm[cold]=CW
    pmix=(F*Wm).sum(1)
    m=~isf_v if VS==2023 else np.ones(nn,bool)
    def sc(z):
        r_=minimize_scalar(lambda c:-bss(IV(z+c)[m],yv[m]),bounds=(-0.3,0.3),method="bounded")
        return -r_.fun
    z0=LG(pmix)
    trm2=season<VS
    def sdev(kk, ent, div):
        v=y[trm2]; k_=kk[trm2]; e_=ent[trm2]
        mu_=v.mean()
        pgg=pd.Series(v).groupby(e_).agg(["sum","count"])
        prr=(pgg["sum"]+300*mu_)/(pgg["count"]+300)
        kgg=pd.Series(v).groupby(k_).agg(["sum","count"])
        kent=(pd.Series(k_)//div).groupby(k_).first()
        base_=prr.reindex(kent.values).to_numpy()
        krt=(kgg["sum"].to_numpy()+300*base_)/(kgg["count"].to_numpy()+300)
        return dict(zip(kgg.index.to_numpy(),krt-base_))
    bidv_all=tr["batter_id"].to_numpy().astype("int64")
    d_p=sdev(kp_c12,pid.astype("int64"),100)
    d_o=sdev(kp_out,pid.astype("int64"),10)
    d_b=sdev(kb_c12,bidv_all,100)
    kv1=kp_c12[va_m]; kv2=kp_out[va_m]; kv3=kb_c12[va_m]
    dv1=np.array([d_p.get(k,0.0) for k in kv1])
    dv2=np.array([d_o.get(k,0.0) for k in kv2])
    dv3=np.array([d_b.get(k,0.0) for k in kv3])
    z0=z0+0.12*dv1-0.10*dv2+0.05*dv3
    base=sc(z0)
    keyv=key[va_m.to_numpy() if hasattr(va_m,'to_numpy') else va_m]
    out=f"VS{VS} 기준 {base:.1f} | "
    dvv=np.array([dev["rev"].get(kk,0.0) for kk in keyv])
    for b in (-1.2,-0.9,-0.6,-0.45,-0.3,-0.15):
        out+=f"β{b:+.2f} {sc(z0+b*dvv)-base:+.2f} | "
    P(out)
