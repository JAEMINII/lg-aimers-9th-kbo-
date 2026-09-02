import os, sys
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = os.path.join(ROOT, "nn_experiments")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "submit_35"))
import features44

def score(p, y):
    r = y.mean()
    return 1e5 * (1.0 - np.mean((p-y)**2) / (r*(1-r)))

def cal(p, shift=-.007, T=1.0279):
    p=np.clip(p,1e-6,1-1e-6)
    z=T*np.log(p/(1-p))+shift
    return 1/(1+np.exp(-z))

def main(vs):
    b=features44.build(os.path.join(ROOT,"open (1)","data"),VS=vs,return_frame=True)
    d=b["frame"]; m=b["m_va"]
    y=d.loc[m,"control_success"].to_numpy(float)
    reg=d.loc[m,"game_type"].to_numpy()==0
    names=["base44","c12_cmh"]
    ps={n:np.load(os.path.join(SC,f"pcgpu{vs}_{n}.npy")) for n in names}
    if vs==2024:
        tab=np.load(os.path.join(SC,"tabm_base_gate.npy"))
        assert len(tab)==len(y), (len(tab),len(y))
        print("TabM mean",tab.mean(),"corr44",np.corrcoef(tab[reg],ps['base44'][reg])[0,1])
        for n,p in ps.items():
            print(n,"cb fixed",score(cal(p[reg]),y[reg]),"best shift",end=" ")
            r=minimize_scalar(lambda s:-score(cal(p[reg],s),y[reg]),bounds=(-.15,.15),method="bounded")
            print(round(-r.fun,3),round(r.x,5))
            vals=[]
            for w in np.arange(0,1.01,.05):
                q=cal((1-w)*tab+w*p)
                vals.append((score(q[reg],y[reg]),w))
            print(" blend fixed best",max(vals),"raw best",max((score(cal((1-w)*tab+w*p,s)[reg],y[reg]),w,s) for w in np.arange(0,1.01,.05) for s in [-.007,-.04,-.02,0]))
        basep, candp = ps["base44"], ps["c12_cmh"]
        mix = []
        for a in np.arange(0, 1.01, .05):
            q = cal((1-a)*basep + a*candp)
            mix.append((score(q[reg], y[reg]), a))
        print("CB-only base->c50 fixed best", max(mix))
    else:
        for n,p in ps.items():
            r=minimize_scalar(lambda s:-score(cal(p[reg],s),y[reg]),bounds=(-.15,.15),method="bounded")
            print(n,"fixed",score(cal(p[reg]),y[reg]),"best",-r.fun,"shift",r.x)

if __name__=='__main__': main(int(sys.argv[1]))
