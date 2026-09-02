# -*- coding: utf-8 -*-
"""56 검증: 55 대조 — 1군 동일성(라우팅 정합), 퓨처스 변화, 시간, 규칙4."""
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
ROOT = os.getcwd()
DATA = "open (1)/data"
cols = list(pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns)
tr = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig")
pool = tr[tr.season == 2024]
rng = np.random.default_rng(0)
pick = np.sort(rng.choice(len(pool), 20000, replace=False))
sub = pool.iloc[pick][cols].copy()
y24 = pool.iloc[pick]["control_success"].to_numpy(float)
PKGS = ("submit_55", "submit_56")
for pkg in PKGS:
    os.makedirs(f"{pkg}/data", exist_ok=True)
    sub.to_csv(f"{pkg}/data/test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"row_id": sub["row_id"], "control_success": 0.5}).to_csv(
        f"{pkg}/data/sample_submission.csv", index=False, encoding="utf-8-sig")
code = ("import runpy,io,contextlib,time;t0=time.time()" + chr(10) +
        "with contextlib.redirect_stdout(io.StringIO()):" + chr(10) +
        "    runpy.run_path('script.py', run_name='__main__')" + chr(10) +
        "print(f'{time.time()-t0:.1f}')" + chr(10))
T, P = {}, {}
for pkg in PKGS:
    r = subprocess.run([sys.executable, "-c", code], cwd=os.path.join(ROOT, pkg),
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=ENV)
    if r.returncode:
        print(pkg, "실패:", r.stderr[-1200:])
        sys.exit(1)
    T[pkg] = float(r.stdout.strip().splitlines()[0])
    g = pd.read_csv(f"{pkg}/output/submission.csv")
    P[pkg] = dict(zip(g["row_id"], g["control_success"]))
    print(f"  {pkg}  {T[pkg]:.1f}s")
rid = sub["row_id"].tolist()
a55 = np.array([P["submit_55"][r] for r in rid])
a56 = np.array([P["submit_56"][r] for r in rid])
est = T["submit_56"] * 396.0 / T["submit_55"] * (6.33 / 6.6)
print("  시간  56=%.1fs 55=%.1fs  ->  56 서버추정 %.2f분  %s" %
      (T["submit_56"], T["submit_55"], est / 60,
       "통과" if est <= 480 else "**초과**"))
isf = sub["game_type"].astype(str).to_numpy() == "F"
d1 = np.abs(a56[~isf] - a55[~isf]).max()
print("  1군 55-56 최대차 %.2e %s" %
      (d1, "(동일해야 함) 통과" if d1 < 1e-9 else "**다름**"))
df = np.abs(a56[isf] - a55[isf])
print("  퓨처스  상관 %.5f  평균|차| %.4f  최대|차| %.4f (달라야 정상)" %
      (np.corrcoef(a55[isf], a56[isf])[0, 1], df.mean(), df.max()))
l = lambda p: np.log(p / (1 - p))
print("  수준  실제 %.5f  55 %.5f  56 %.5f  Dz %.5f" %
      (y24.mean(), a55.mean(), a56.mean(), l(a56.mean()) - l(y24.mean())))
full = pd.read_csv("submit_56/data/test.csv", encoding="utf-8-sig")
per = rng.permutation(len(full))
full.iloc[per].to_csv("submit_56/data/test.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"row_id": full["row_id"].to_numpy()[per],
              "control_success": 0.5}).to_csv(
    "submit_56/data/sample_submission.csv", index=False, encoding="utf-8-sig")
subprocess.run([sys.executable, "-c", code], cwd=os.path.join(ROOT, "submit_56"),
               capture_output=True, text=True, encoding="utf-8",
               errors="replace", env=ENV)
g = pd.read_csv("submit_56/output/submission.csv")
d2 = float(max(abs(P["submit_56"][r] - v)
               for r, v in zip(g["row_id"], g["control_success"])))
print("  규칙4 순서 %.2e %s" % (d2, "통과" if d2 < 1e-6 else "**위반**"))
for p in PKGS:
    shutil.rmtree(f"{p}/data", ignore_errors=True)
    shutil.rmtree(f"{p}/output", ignore_errors=True)
    shutil.rmtree(f"{p}/__pycache__", ignore_errors=True)
