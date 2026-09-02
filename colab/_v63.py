# -*- coding: utf-8 -*-
"""63 검증: 54 대조 — 보정 정확성(로짓차 == T·β·Δ), 커버 0행 동일성, 시간, 규칙4."""
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
ROOT = os.getcwd()
DATA = "open (1)/data"
T54_SERVER_MIN = 6.33
CALIB_T = 1.04
BETA = 0.12
cols = list(pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns)
tr = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig")
pool = tr[tr.season == 2024]
rng = np.random.default_rng(0)
pick = np.sort(rng.choice(len(pool), 20000, replace=False))
sub = pool.iloc[pick][cols].copy()
PKGS = ("submit_54", "submit_63")
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
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=ENV)
    if r.returncode:
        print(pkg, "실패:", r.stderr[-1500:])
        sys.exit(1)
    t = float(r.stdout.strip().splitlines()[0])
    g = pd.read_csv(f"{pkg}/output/submission.csv")
    return t, dict(zip(g["row_id"], g["control_success"]))


T, P = {}, {}
for pkg in PKGS:
    T[pkg], P[pkg] = run(pkg)
    print(f"  {pkg}  {T[pkg]:.1f}s")
rid = sub["row_id"].tolist()
a54 = np.array([P["submit_54"][r] for r in rid])
a63 = np.array([P["submit_63"][r] for r in rid])
est = T["submit_63"] / T["submit_54"] * T54_SERVER_MIN
print("  시간  63=%.1fs 54=%.1fs  ->  63 서버추정 %.2f분  %s" %
      (T["submit_63"], T["submit_54"], est,
       "통과" if est <= 8.0 else "**초과**"))

pz = np.load("submit_63/model/pcnt.npz", allow_pickle=False)
mp = {(int(a), int(b)): float(c) for a, b, c in
      zip(pz["pitcher_id"], pz["cnt"], pz["delta"])}
cnt = sub["balls_before"].to_numpy(int) * 3 + sub["strikes_before"].to_numpy(int)
dv = np.array([mp.get((int(p), int(c)), 0.0)
               for p, c in zip(sub["pitcher_id"], cnt)])
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9)
                      / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
diff = lg(a63) - lg(a54)
exp = CALIB_T * BETA * dv
err = np.abs(diff - exp).max()
z0 = dv == 0
d0 = np.abs(a63[z0] - a54[z0]).max() if z0.any() else 0
print("  보정정확성  커버 %.3f  로짓차-기대 최대오차 %.2e %s" %
      (np.mean(dv != 0), err, "통과" if err < 1e-6 else "**불일치**"))
print("  비커버 동일성 %.2e %s" % (d0, "통과" if d0 < 1e-12 else "**다름**"))

full = pd.read_csv("submit_63/data/test.csv", encoding="utf-8-sig")
per = rng.permutation(len(full))
full.iloc[per].to_csv("submit_63/data/test.csv", index=False,
                      encoding="utf-8-sig")
pd.DataFrame({"row_id": full["row_id"].to_numpy()[per],
              "control_success": 0.5}).to_csv(
    "submit_63/data/sample_submission.csv", index=False, encoding="utf-8-sig")
_, Pp = run("submit_63")
d2 = float(max(abs(P["submit_63"][r] - v) for r, v in Pp.items()))
print("  규칙4 순서 %.2e %s" % (d2, "통과" if d2 < 1e-6 else "**위반**"))
for p in PKGS:
    shutil.rmtree(f"{p}/data", ignore_errors=True)
    shutil.rmtree(f"{p}/output", ignore_errors=True)
    shutil.rmtree(f"{p}/__pycache__", ignore_errors=True)
print("검증 끝")
