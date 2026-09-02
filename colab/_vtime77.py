# -*- coding: utf-8 -*-
import os, shutil, subprocess, sys, time
import numpy as np, pandas as pd
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
ROOT = os.getcwd(); DATA = "open (1)/data"
cols = list(pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns)
tr = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig")
pool = tr[tr.season == 2024]
rng = np.random.default_rng(0)
pick = np.sort(rng.choice(len(pool), 20000, replace=False))
sub = pool.iloc[pick][cols].copy()
PKGS = ("submit_71", "submit_76", "submit_77", "submit_76", "submit_77", "submit_71")
for pkg in PKGS:
    os.makedirs(f"{pkg}/data", exist_ok=True)
    sub.to_csv(f"{pkg}/data/test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"row_id": sub["row_id"], "control_success": 0.5}).to_csv(
        f"{pkg}/data/sample_submission.csv", index=False, encoding="utf-8-sig")
code = ("import runpy,io,contextlib,time;t0=time.time()" + chr(10) +
        "with contextlib.redirect_stdout(io.StringIO()):" + chr(10) +
        "    runpy.run_path('script.py', run_name='__main__')" + chr(10) +
        "print(f'{time.time()-t0:.1f}')" + chr(10))
T = {}
for pkg in PKGS:
    r = subprocess.run([sys.executable, "-c", code], cwd=os.path.join(ROOT, pkg),
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV)
    if r.returncode: print(pkg, "실패:", r.stderr[-500:]); sys.exit(1)
    T[pkg+str(len(T))] = float(r.stdout.strip().splitlines()[0])
    print(f"  {pkg}  {T[pkg]:.1f}s  (71대비 x{T[pkg]/1:.3f}  서버추정 {T[pkg]/T['submit_71']*5.54:.2f}분)")
for p in PKGS:
    for d in ("data", "output", "__pycache__"):
        shutil.rmtree(f"{p}/{d}", ignore_errors=True)
print("끝")
