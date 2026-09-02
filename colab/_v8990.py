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
PKGS = ("submit_71", "submit_81", "submit_83", "submit_89", "submit_90")
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
    if r.returncode: print(pkg, "실패:", r.stderr[-1200:], flush=True); sys.exit(1)
    t = float(r.stdout.strip().splitlines()[0])
    g = pd.read_csv(f"{pkg}/output/submission.csv")
    return t, dict(zip(g["row_id"], g["control_success"]))
T, P = {}, {}
for pkg in PKGS:
    T[pkg], P[pkg] = run(pkg); print(f"  {pkg}  {T[pkg]:.1f}s", flush=True)
rid = sub["row_id"].tolist()
A = {p: np.array([P[p][r] for r in rid]) for p in PKGS}
for pk in ("submit_89", "submit_90"):
    e = T[pk] / T["submit_71"] * 5.54
    print("  시간 %s %.2f분 %s" % (pk, e, "통과" if e <= 8.0 else "**초과**"), flush=True)
t13p = ((sub["pitcher_team_id"].to_numpy() == 13) & (sub["game_type"] == "R").to_numpy())
gm = sub["game_month"].to_numpy()
for pk, ref in (("submit_89", "submit_81"), ("submit_90", "submit_83")):
    c = np.corrcoef(A[ref], A[pk])[0, 1]
    d = A[pk] - A[ref]
    e34 = d[t13p & (gm <= 4)].mean(); e56 = d[t13p & (gm >= 5) & (gm <= 6)].mean(); e7 = d[t13p & (gm >= 7)].mean()
    print("  %s 상관(%s) %.4f  수준 %.5f  13투Δ 3-4월 %+0.5f / 5-6월 %+0.5f / 7월+ %+0.5f" % (
        pk, ref, c, A[pk].mean(), e34, e56, e7), flush=True)
full = pd.read_csv("submit_89/data/test.csv", encoding="utf-8-sig")
per = rng.permutation(len(full))
full.iloc[per].to_csv("submit_89/data/test.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"row_id": full["row_id"].to_numpy()[per], "control_success": 0.5}).to_csv(
    "submit_89/data/sample_submission.csv", index=False, encoding="utf-8-sig")
_, Pp = run("submit_89")
d2 = float(max(abs(P["submit_89"][r] - v) for r, v in Pp.items()))
print("  규칙4 순서(89) %.2e %s" % (d2, "통과" if d2 < 1e-6 else "**위반**"), flush=True)
for p in PKGS:
    for d_ in ("data", "output", "__pycache__"):
        shutil.rmtree(f"{p}/{d_}", ignore_errors=True)
print("검증 끝", flush=True)
