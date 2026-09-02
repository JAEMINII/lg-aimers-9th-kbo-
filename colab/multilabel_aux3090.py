"""Multi-label reverse/middle auxiliary TabM probe on the 3090.

The reverse and middle outcomes are not mutually exclusive in the source
data, so this is the less restrictive alternative to multistate_softmax.py.
Only the direct success head is reported for selection.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT=Path('/root'); DATA=Path(os.environ.get('AIMERS_DATA','/root/data'))
OUT=Path(os.environ.get('AIMERS_OUT','/root/multilabel_out')); OUT.mkdir(parents=True,exist_ok=True)
VS=int(os.environ.get('VS','2024')); SEEDS=tuple(int(x) for x in os.environ.get('SEEDS','42,1,777').split(','))
EPOCHS=int(os.environ.get('EPOCHS','2')); BS=int(os.environ.get('BS','2048')); K=int(os.environ.get('K','32')); DBLOCK=int(os.environ.get('DBLOCK','512'))
LR=float(os.environ.get('LR','0.003')); AUX_W=float(os.environ.get('AUX_W','0.3'))
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); AMP=DEVICE.type=='cuda'
sys.path.insert(0,str(ROOT)); import features44 as FF; import tabm_gate_gpu as G
from rtdl_num_embeddings import LinearReLUEmbeddings
from tabm import TabM
from train_chan_3 import preprocess as PP

def log(s):
 line=f'[{time.strftime("%H:%M:%S")}] {s}'; print(line,flush=True); (OUT/'progress.txt').open('a').write(line+'\n')
def bss(p,y):
 r=y.mean(); return 1e5*(1-((p-y)**2).mean()/(r*(1-r)))
def shift(p,c):
 q=np.clip(p,1e-6,1-1e-6); z=np.log(q/(1-q)); return 1/(1+np.exp(-(z+c)))
def best(p,y):
 cs=np.linspace(-.12,.12,481); v=[bss(shift(p,c),y) for c in cs]; i=int(np.argmax(v)); return float(v[i]),float(cs[i])

def recover_aux(df):
 pid=df.pitcher_id.to_numpy(); n=df.asof_pitcher_n.fillna(0.).to_numpy(float)
 nxt=(pid[1:]==pid[:-1]) & np.isclose(np.diff(n),1.,atol=1e-8); src=np.flatnonzero(nxt)+1; dst=src-1
 out=[]
 for col in ['asof_pitcher_reverse_rate','asof_pitcher_middle_rate']:
  cum=df[col].fillna(0.).to_numpy(float)*n; inc=cum[src]-cum[dst]; lab=np.rint(inc)
  good=(np.abs(inc-lab)<.25)&((lab==0)|(lab==1)); v=np.full(len(df),np.nan); v[dst[good]]=lab[good]; out.append(v)
 return np.stack(out,1).astype(np.float32)

def make_model(n_num,cards,seed):
 torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
 return TabM.make(n_num_features=n_num,cat_cardinalities=[int(c) for c in cards],d_out=3,num_embeddings=LinearReLUEmbeddings(n_num,d_embedding=16),arch_type='tabm',k=K,n_blocks=3,d_block=DBLOCK,dropout=.1).to(DEVICE)

def train_model(model,idx,y,aux,w,seed):
 torch.manual_seed(seed); ii=torch.from_numpy(idx.astype(np.int64)); yy=torch.from_numpy(y.astype(np.float32)); aa=torch.from_numpy(aux); ww=torch.from_numpy(w.astype(np.float32)); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=3e-4); model.train()
 for ep in range(EPOCHS):
  perm=torch.randperm(len(idx)); total=0.
  for st in range(0,len(idx),BS):
   pos=perm[st:st+BS]; b=ii[pos]; xn=G.XN[b].to(DEVICE); xc=G.XC[b].to(DEVICE); yb=yy[b].to(DEVICE); ab=aa[b].to(DEVICE); wb=ww[b].to(DEVICE)
   opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=AMP):
    raw=model(xn,xc); dl=raw[:,:,0]; dm=dl.mean(1); dp=torch.sigmoid(dm); direct=((.35*F.binary_cross_entropy_with_logits(dm,yb,reduction='none')+.25*(dp-yb).square())*wb).sum()/wb.sum(); loss=direct
    for j in range(2):
     m=torch.isfinite(ab[:,j])
     if bool(m.any()):
      z=raw[m,:,j+1]; t=ab[m,j,None].expand_as(z); ce=F.binary_cross_entropy_with_logits(z,t,reduction='none').mean(1); loss=loss+AUX_W*(ce*wb[m]).sum()/wb[m].sum()
   loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); total+=float(loss.detach())*len(b)
  log(f'VS{VS} s{seed} ep{ep+1}/{EPOCHS} loss={total/len(idx):.6f}')
 return model

@torch.no_grad()
def predict(model,idx):
 model.eval(); ii=torch.from_numpy(idx.astype(np.int64)); out=[]
 for st in range(0,len(idx),8192):
  b=ii[st:st+8192]; xn=G.XN[b].to(DEVICE); xc=G.XC[b].to(DEVICE); raw=model(xn,xc); out.append(torch.sigmoid(raw[:,:,0]).mean(1).float().cpu().numpy())
 return np.concatenate(out)

def main():
 raw=PP.sort_by_row_id(pd.read_csv(DATA/'train.csv',encoding='utf-8-sig')); aux=recover_aux(raw); y=raw.control_success.to_numpy(np.float32); season=raw.season.to_numpy(np.int16); isf=raw.game_type.astype(str).to_numpy()=='F'
 built=FF.build(str(DATA),VS=VS); X44=built['X44'].astype(np.float32); names=list(built['F44']); old=season<=2022; c4=np.where(old&isf,0.,np.where(old&~isf,1.,np.where(isf,2.,3.))).astype(np.float32)[:,None]; ci=[names.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]; Xin=np.concatenate([X44,c4],1); tr=(season<VS); va=(season==VS); Xn,Xc,cards=G.prep(Xin,tr,ci+[X44.shape[1]]); G.XN,G.XC=torch.from_numpy(Xn),torch.from_numpy(Xc); ti=np.flatnonzero(tr); vi=np.flatnonzero(va); w=np.ones(len(y),np.float32); w[ti]=np.where(isf[ti]&old[ti],.1,1.); log(f'X={Xin.shape} train={len(ti):,} val={len(vi):,} aux={np.isfinite(aux[ti]).any(1).sum():,} K={K} d={DBLOCK} aw={AUX_W}')
 preds=[]
 for s in SEEDS:
  m=train_model(make_model(Xn.shape[1],cards,s),ti,y,aux,w,s); p=predict(m,vi); preds.append(p); np.save(OUT/f'aux_vs{VS}_s{s}.npy',p); del m; torch.cuda.empty_cache()
 p=np.mean(preds,0); yv=y[vi].astype(float); fv=isf[vi]; rep={'overall':best(p,yv),'regular':best(p[~fv],yv[~fv]),'futures':best(p[fv],yv[fv])}; log(json.dumps(rep)); (OUT/'report.json').write_text(json.dumps(rep,indent=2))
if __name__=='__main__': main()
