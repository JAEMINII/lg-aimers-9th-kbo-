import os
import sys
import numpy as np

sys.path.insert(0, "/root")

for VS in (2022, 2024):
    os.environ["VS"] = str(VS)
    os.environ["AIMERS_DATA"] = "/root/open (1)/data"
    os.environ["AIMERS_OUT"] = f"/root/modelcheck/vs{VS}/arch"
    for name in ("tabm_gate_gpu", "features44"):
        sys.modules.pop(name, None)
    import tabm_gate_gpu as G
    from features44 import best_shift
    y = G.yv.astype(float)
    reg = ~G.is_f[G.gate]
    cb = np.asarray(G.CB, dtype=float)
    fm = np.load(f"/root/modelcheck/deepfm/vs{VS}/deepfm_vs{VS}_logit.npy")
    tabs = [np.load(f"/root/modelcheck/vs{VS}/arch/ar_tabm_s{s}.npy")
            for s in (42, 1, 777, 2)]
    tab = np.mean(tabs, axis=0)
    print(f"VS={VS} shapes y={y.shape} cb={cb.shape} tab={tab.shape} fm={fm.shape}", flush=True)
    for wt in (0.3, 0.5, 0.7):
        base = wt * tab + (1.0 - wt) * cb
        sb = best_shift(base[reg], y[reg])[0]
        bl = np.log(np.clip(base, 1e-6, 1 - 1e-6) /
                    (1 - np.clip(base, 1e-6, 1 - 1e-6)))
        vals = []
        for a in (0.1, 0.2, 0.3, 0.4):
            p = 1.0 / (1.0 + np.exp(-np.clip(bl + a * (fm - bl), -30, 30)))
            vals.append((a, best_shift(p[reg], y[reg])[0] - sb))
        print(f"  tab_weight={wt:.1f} base={sb:.1f} "
              f"deltas=" + ",".join(f"{a:.1f}:{d:+.1f}" for a, d in vals), flush=True)
