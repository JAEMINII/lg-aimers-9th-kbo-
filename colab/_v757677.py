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
y24 = pool.iloc[pick]["control_success"].to_numpy(float)
PKGS = ("submit_74", "submit_75", "submit_76", "submit_77")
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
est = max(T[p] for p in PKGS[1:]) / T["submit_74"] * 5.54
print("  시간  서버추정 %.2f분  %s" % (est, "통과" if est <= 8.0 else "**초과**"))
for p in PKGS[1:]:
    c = np.corrcoef(A["submit_74"], A[p])[0, 1]
    print("  %s 상관74 %.4f  수준 %.5f (74 %.5f  실제 %.5f)" %
          (p, c, A[p].mean(), A["submit_74"].mean(), y24.mean()))
t13p = ((sub["pitcher_team_id"].to_numpy() == 13) & (sub["game_type"] == "R").to_numpy())
t13b = ((sub["batter_team_id"].to_numpy() == 13) & (sub["game_type"] == "R").to_numpy())
oth = ~t13p & ~t13b
d65 = A["submit_76"] - A["submit_75"]
print("  발화 76: 13투 변화율 %.3f  평균Δ %+0.5f | 비13 변화율 %.3f" %
      ((np.abs(d65[t13p]) > 1e-9).mean(), d65[t13p].mean(), (np.abs(d65[oth]) > 1e-9).mean()))
d75 = A["submit_77"] - A["submit_75"]
print("  발화 77: 13투 평균Δ %+0.5f  13타 평균Δ %+0.5f  비13 변화율 %.3f" %
      (d75[t13p].mean(), d75[t13b].mean(), (np.abs(d75[oth]) > 1e-9).mean()))
full = pd.read_csv("submit_76/data/test.csv", encoding="utf-8-sig")
per = rng.permutation(len(full))
full.iloc[per].to_csv("submit_76/data/test.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"row_id": full["row_id"].to_numpy()[per], "control_success": 0.5}).to_csv(
    "submit_76/data/sample_submission.csv", index=False, encoding="utf-8-sig")
_, Pp = run("submit_76")
d2 = float(max(abs(P["submit_76"][r] - v) for r, v in Pp.items()))
print("  규칙4 순서(76) %.2e %s" % (d2, "통과" if d2 < 1e-6 else "**위반**"))
for p in PKGS:
    for d in ("data", "output", "__pycache__"):
        shutil.rmtree(f"{p}/{d}", ignore_errors=True)
print("검증 끝")
