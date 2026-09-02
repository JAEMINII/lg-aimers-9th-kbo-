# -*- coding: utf-8 -*-
"""롤링 학습 CB(<=2024H1)의 2024H2 잔차 — 13투/13타/비13, 월별."""
import sys, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
def P(s): sys.stdout.buffer.write((s+"\n").encode("utf-8"))
def bss(p,t):
    r=t.mean(); return 100000*(1-((p-t)**2).mean()/(r*(1-r)))
def LG(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def IV(z): return 1/(1+np.exp(-z))
tr=pd.read_csv("open (1)/data/train.csv",encoding="utf-8-sig",
    usecols=["row_id","season","game_month","game_type",
             "pitcher_team_id","batter_team_id","control_success"])
rid=tr["row_id"].str.slice(6).astype("int64")
tr=tr.assign(_r=rid).sort_values("_r").reset_index(drop=True)
idx=np.load("colab/_dl/cbh7_A_meta.npy").ravel()
p=np.load("colab/_dl/cbh7_A_b58.npy").astype(np.float64)
pt13=np.load("colab/_dl/cbh7_A_t13.npy").astype(np.float64)
va=tr.iloc[idx]
P(f"검증행 {len(idx)}  시즌 {sorted(va.season.unique())}  월 {sorted(va.game_month.unique())}")
yv=va.control_success.to_numpy(float)
isf=va.game_type.astype(str).to_numpy()=="F"
m=~isf
r0=minimize_scalar(lambda c:-bss(IV(LG(p[m])+c),yv[m]),bounds=(-0.3,0.3),method="bounded")
pz=IV(LG(p)+r0.x)
r1=minimize_scalar(lambda c:-bss(IV(LG(pt13[m])+c),yv[m]),bounds=(-0.3,0.3),method="bounded")
pz13=IV(LG(pt13)+r1.x)
pt=va.pitcher_team_id.to_numpy(); bt=va.batter_team_id.to_numpy()
mo=va.game_month.to_numpy()
p13=(pt==13)&(bt!=13); b13=(bt==13)&(pt!=13); n13=(pt!=13)&(bt!=13)
P("== 롤링CB(b58, <=24H1 학습) R행 잔차")
for tag,mask in (("13투",p13),("13타",b13),("비13",n13)):
    mm=mask&m
    P(f"  {tag}  n={int(mm.sum())}  잔차 {100*(yv[mm].mean()-pz[mm].mean()):+.2f}pp")
for M in sorted(set(mo[m])):
    row=f"  {int(M):02d}월 "
    for tag,mask in (("13투",p13),("13타",b13)):
        mm=mask&m&(mo==M)
        if mm.sum()<500: row+=f" {tag} n소 "; continue
        row+=f" {tag} {100*(yv[mm].mean()-pz[mm].mean()):+.2f} "
    P(row)
P("== t13 지시자 포함판(t13팔)의 같은 잔차 (이중계상 확인)")
for tag,mask in (("13투",p13),("13타",b13)):
    mm=mask&m
    P(f"  {tag}  잔차 {100*(yv[mm].mean()-pz13[mm].mean()):+.2f}pp")
P("== 참고: F행 t13")
mmf=(p13|b13)&isf
if mmf.sum()>500:
    P(f"  t13∩F n={int(mmf.sum())}  잔차 {100*(yv[mmf].mean()-pz[mmf].mean()):+.2f}pp")
