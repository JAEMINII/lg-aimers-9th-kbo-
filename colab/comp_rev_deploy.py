# -*- coding: utf-8 -*-
"""배치용 rev lookup — 전체 train(<=2024) as-of 차분 복원 + (투수,타자손,카운트12) 계층수축 편차."""
import sys, numpy as np, pandas as pd
use=["row_id","season","game_type","pitcher_id","batter_hand",
     "balls_before","strikes_before","control_success","asof_pitcher_n",
     "asof_pitcher_reverse_rate"]
tr=pd.read_csv("data/train.csv",encoding="utf-8-sig",usecols=use)
rid=tr["row_id"].str.slice(6).astype("int64")
tr=tr.assign(_r=rid).sort_values("_r").reset_index(drop=True)
n=tr["asof_pitcher_n"].fillna(0).to_numpy(float)
crev=tr["asof_pitcher_reverse_rate"].fillna(0).to_numpy(float)*n
pid=tr["pitcher_id"].to_numpy()
ordidx=np.arange(len(tr))
nxt=pd.DataFrame({"pid":pid,"i":ordidx}).groupby("pid")["i"].shift(-1).to_numpy()
ok=~np.isnan(nxt)
nx=nxt[ok].astype(int); cur=ordidx[ok]
dn=n[nx]-n[cur]
rev=np.full(len(tr),np.nan)
good=np.abs(dn-1)<1e-6
rev[cur[good]]=np.round(crev[nx[good]]-crev[cur[good]])
val=(~np.isnan(rev))&(np.isin(rev,[0,1]))
print(f"복원 커버 {val.mean():.4f}  reverse율 {np.nanmean(rev[val]):.4f}", flush=True)
c12=(tr["balls_before"].to_numpy()*3+tr["strikes_before"].to_numpy()).astype("int64")
bh=tr["batter_hand"].to_numpy().astype("int64")
key=pid.astype("int64")*1000+bh*100+c12
v=rev[val]; k_=key[val]; e_=pid.astype("int64")[val]
mu=v.mean()
pg=pd.Series(v).groupby(e_).agg(["sum","count"])
pr=(pg["sum"]+300*mu)/(pg["count"]+300)
kg=pd.Series(v).groupby(k_).agg(["sum","count"])
kent=(pd.Series(k_)//1000).groupby(k_).first()
base=pr.reindex(kent.values).to_numpy()
krt=(kg["sum"].to_numpy()+300*base)/(kg["count"].to_numpy()+300)
keys=kg.index.to_numpy().astype("int64")
dev=(krt-base).astype("float32")
np.savez_compressed("/root/comp_rev.npz", keys=keys, dev=dev)
print(f"저장 {len(keys)}키  dev 표준편차 {dev.std():.5f}  범위 [{dev.min():.4f},{dev.max():.4f}]", flush=True)
