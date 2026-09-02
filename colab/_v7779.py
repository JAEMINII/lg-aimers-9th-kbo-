# -*- coding: utf-8 -*-
import os, shutil, subprocess, sys
import numpy as np, pandas as pd
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
ROOT = os.getcwd(); DATA = "open (1)/data"
cols = list(pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns)
tr = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig")
pool = tr[tr.season == 2024]
rng = np.random.default_rng(0)
pick = np.sort(rng.choice(len(pool), 20000, replace=False))
sub = pool.iloc[pick][cols].copy()
PKGS = ("submit_71", "submit_77p", "submit_79")
for pkg in PKGS:
    os.makedirs(f"{pkg}/data", exist_ok=True)
    sub.to_csv(f"{pkg}/data/test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"row_id": sub["row_id"], "control_success": 0.5}).to_csv(
        f"{pkg}/data/sample_submission.csv", index=False, encoding="utf-8-sig")
code = ("import runpy,io,contextlib,time;t0=time.time()" + chr(10) +
        "with contextlib.redirect_stdout(io.StringIO()):" + chr(10) +
        "    runpy.run_path('script.py', run_name='__main__')" + chr(10) +
        "print(f'{time.time()-t0:.1f}')" + chr(10))
def run(pkg):
    r = subprocess.run([sys.executable, "-c", code], cwd=os.path.join(ROOT, pkg),
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV)
    if r.returncode: print(pkg, "실패:", r.stderr[-1200:]); sys.exit(1)
    t = float(r.stdout.strip().splitlines()[0])
    g = pd.read_csv(f"{pkg}/output/submission.csv")
    return t, dict(zip(g["row_id"], g["control_success"]))
T, P = {}, {}
for pkg in PKGS:
    T[pkg], P[pkg] = run(pkg); print(f"  {pkg}  {T[pkg]:.1f}s")
rid = sub["row_id"].tolist()
A = {p: np.array([P[p][r] for r in rid]) for p in PKGS}
print("  시간 78 서버추정 %.2f분  %s" % (T["submit_79"]/T["submit_71"]*5.54,
      "통과" if T["submit_79"]/T["submit_71"]*5.54 <= 8.0 else "**초과**"))
c = np.corrcoef(A["submit_77p"], A["submit_79"])[0, 1]
print("  스모크 상관(75) %.4f  수준 78 %.5f (75 %.5f)" % (c, A["submit_79"].mean(), A["submit_77p"].mean()))
d = A["submit_79"] - A["submit_77p"]
print("  발화: 변화율 %.3f  |Δ|중앙 %.5f  최대 %.5f" % ((np.abs(d)>1e-9).mean(), np.median(np.abs(d[np.abs(d)>1e-9])), np.abs(d).max()))
full = pd.read_csv("submit_79/data/test.csv", encoding="utf-8-sig")
per = rng.permutation(len(full))
full.iloc[per].to_csv("submit_79/data/test.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"row_id": full["row_id"].to_numpy()[per], "control_success": 0.5}).to_csv(
    "submit_79/data/sample_submission.csv", index=False, encoding="utf-8-sig")
_, Pp = run("submit_79")
d2 = float(max(abs(P["submit_79"][r] - v) for r, v in Pp.items()))
print("  규칙4 순서(78) %.2e %s" % (d2, "통과" if d2 < 1e-6 else "**위반**"))
for p in PKGS:
    for d_ in ("data", "output", "__pycache__"):
        shutil.rmtree(f"{p}/{d_}", ignore_errors=True)
print("검증 끝")
