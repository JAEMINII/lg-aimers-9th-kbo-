"""Train the selected multi-state TabM on all rows and export NumPy archives."""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path('/root'); DATA=Path(os.environ.get('AIMERS_DATA','/root/data')); OUT=Path(os.environ.get('AIMERS_OUT','/root/ms_export'))
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'aimers'))
import features44 as FF
import multistate_softmax as M
import tabm_gate_gpu as G
from train_chan_3 import preprocess as PP

def recover_state(df):
 pid=df.pitcher_id.to_numpy(); n=df.asof_pitcher_n.fillna(0.).to_numpy(float)
 nxt=(pid[1:]==pid[:-1]) & np.isclose(np.diff(n),1.,atol=1e-8); src=np.flatnonzero(nxt)+1; dst=src-1
 vals={}
 for name,col in [('reverse','asof_pitcher_reverse_rate'),('middle','asof_pitcher_middle_rate')]:
  cum=df[col].fillna(0.).to_numpy(float)*n; inc=cum[src]-cum[dst]; lab=np.rint(inc); good=(np.abs(inc-lab)<.25)&((lab==0)|(lab==1)); v=np.full(len(df),np.nan); v[dst[good]]=lab[good]; vals[name]=v
 y=df.control_success.to_numpy(float); s=np.full(len(df),-1,np.int64); s[y>.5]=0; known=np.isfinite(vals['reverse'])&np.isfinite(vals['middle']); s[known&(y<=.5)&(vals['reverse']>.5)]=1; s[known&(y<=.5)&(vals['reverse']<=.5)&(vals['middle']>.5)]=2; s[known&(y<=.5)&(vals['reverse']<=.5)&(vals['middle']<=.5)]=3
 return s

def main():
 raw=PP.sort_by_row_id(pd.read_csv(DATA/'train.csv',encoding='utf-8-sig')); y=raw.control_success.to_numpy(np.float32); season=raw.season.to_numpy(np.int16); isf=raw.game_type.astype(str).to_numpy()=='F'; old=season<=2022
 aux=recover_state(raw); built=FF.build(str(DATA),VS=2025); X44=built['X44'].astype(np.float32); names=list(built['F44']); c4=np.where(old&isf,0.,np.where(old&~isf,1.,np.where(isf,2.,3.))).astype(np.float32)[:,None]; Xin=np.concatenate([X44,c4],1); ci=[names.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]; mask=np.ones(len(y),bool); Xn,Xc,cards=G.prep(Xin,mask,ci+[X44.shape[1]]); G.XN,G.XC=torch.from_numpy(Xn),torch.from_numpy(Xc); idx=np.arange(len(y),dtype=np.int64); w=np.where(isf&old,.1,1.).astype(np.float32); OUT.mkdir(parents=True,exist_ok=True)
 # Metadata is in the same format consumed by train_chan_3/submission_script.py.
 ni=[j for j in range(Xin.shape[1]) if j not in ci+[X44.shape[1]]]; catcols=PP.TABM_CATEGORICAL_FEATURES+['abs_regime']; numcols=[names[j] for j in ni]; catvals={c:sorted(set(Xin[:,j][np.isfinite(Xin[:,j])].tolist())) for c,j in zip(catcols,ci+[X44.shape[1]])}; med=np.nanmedian(Xin[:,ni],0); med=np.where(np.isfinite(med),med,0.); filled=np.where(np.isnan(Xin[:,ni]),med,Xin[:,ni]); mu=filled.mean(0); sd=filled.std(0)+1e-6; missing=[numcols[k] for k,j in enumerate(ni) if np.isnan(Xin[:,j]).any()]
 prep={'feature_names':names+['abs_regime'],'cat_cols':catcols,'num_cols':numcols,'cat_values':catvals,'medians':dict(zip(numcols,med.tolist())),'means':dict(zip(numcols,mu.tolist())),'stds':dict(zip(numcols,sd.tolist())),'missing_cols':missing}
 G.log(f'export train rows={len(idx):,} X={Xin.shape} K={M.K} d={M.DBLOCK} epochs={M.EPOCHS}')
 for seed in M.SEEDS:
  m=M.train_model(M.make_model(Xn.shape[1],cards,seed),idx,y,aux,w,seed,f'EXPORT s{seed}'); state={k:v.detach().cpu() for k,v in m.state_dict().items()}; payload={'config':{'k':M.K,'arch_type':'tabm','n_blocks':3,'d_block':M.DBLOCK,'num_embeddings':'linear_relu','loss':'multistate_direct_aux'},'preprocessor':prep,'model_state':state}; path=OUT/f'ms_seed{seed}.pt'; torch.save(payload,path); M.log(f'saved {path}'); del m; torch.cuda.empty_cache()
 print(json.dumps({'out':str(OUT),'rows':len(idx),'features':len(prep['feature_names'])},ensure_ascii=False))
if __name__=='__main__': main()
