"""Canonical-preprocess version of the multi-state TabM probe.

The first probe used features44 for both train and validation.  Its
season-specific plat_dev can differ from the deploy-time history lookup for
2025, so this version trains on the same PP.transform_features contract used
by submission inference.
"""
from multistate_softmax import *

def main_canonical():
    raw = PP.sort_by_row_id(pd.read_csv(DATA / "train.csv", encoding="utf-8-sig"))
    aux_state = recover_state(raw)
    y = raw.control_success.to_numpy(np.float32)
    season = raw.season.to_numpy(np.int16)
    is_f = raw.game_type.astype(str).to_numpy() == "F"
    all_report = {}
    for vs in FOLDS:
        t0=time.time()
        hist=PP.fit_history_tables(raw[season < vs])
        Xbase=PP.transform_features(raw, hist, train_mode=True)
        names=list(Xbase.columns)
        old=season<=2022
        c4=np.where(old&is_f,0.,np.where(old&~is_f,1.,np.where(is_f,2.,3.))).astype(np.float32)[:,None]
        ci=[names.index(c) for c in PP.TABM_CATEGORICAL_FEATURES]
        Xin=np.concatenate([Xbase.to_numpy(np.float32),c4],1)
        train_mask=(season<vs)&(season>=MIN_SEASON)
        if BRANCH=='regular': train_mask &= ~is_f
        elif BRANCH=='futures': train_mask &= is_f
        Xn,Xc,cards=G.prep(Xin,train_mask,ci+[Xbase.shape[1]])
        G.XN,G.XC,G.cards=torch.from_numpy(Xn),torch.from_numpy(Xc),cards
        tr_idx,va_idx=np.flatnonzero(train_mask),np.flatnonzero(season==vs)
        w=np.ones(len(y),np.float32); w[tr_idx]=np.where(is_f[tr_idx]&old[tr_idx],.1,1.)
        log(f"CANON VS={vs} X={Xin.shape} train={len(tr_idx):,} val={len(va_idx):,} states={int(np.sum(aux_state[tr_idx]>=0)):,}/{len(tr_idx):,} K={K} d={DBLOCK}")
        pdirect=[]; pstate=[]
        for seed in SEEDS:
            m=train_model(make_model(Xn.shape[1],cards,seed),tr_idx,y,aux_state,w,seed,f"CANON VS{vs} s{seed}")
            a,b=predict(m,va_idx); pdirect.append(a); pstate.append(b); del m; torch.cuda.empty_cache()
        yv=y[va_idx].astype(float); fv=is_f[va_idx]; report={}
        for name,arrs in (("direct",pdirect),("state",pstate),("mix25",[.75*a+.25*b for a,b in zip(pdirect,pstate)])):
            p=np.mean(arrs,0); report[name]={"overall":best_shift(p,yv)[0],"shift":best_shift(p,yv)[1],"regular":best_shift(p[~fv],yv[~fv])[0],"futures":best_shift(p[fv],yv[fv])[0]}; np.save(OUT/f"canon_vs{vs}_{name}.npy",p)
        all_report[str(vs)]={"scores":report,"seconds":time.time()-t0}; log(json.dumps({"CANON_VS":vs,**report}))
    (OUT/"report.json").write_text(json.dumps(all_report,indent=2))

if __name__=='__main__': main_canonical()
