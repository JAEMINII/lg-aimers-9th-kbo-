# -*- coding: utf-8 -*-
"""시간만 재측정: 44/53 교차 2회씩 — 앵커 잡음 제거."""
import os, shutil, subprocess, sys
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
import numpy as np, pandas as pd
ROOT = os.getcwd(); DATA = "open (1)/data"
cols = list(pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns)
tr = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig")
pool = tr[tr.season == 2024]
rng = np.random.default_rng(0)
pick = np.sort(rng.choice(len(pool), 20000, replace=False))
sub = pool.iloc[pick][cols].copy()
for pkg in ("submit_44", "submit_53"):
    os.makedirs(f"{pkg}/data", exist_ok=True)
    sub.to_csv(f"{pkg}/data/test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"row_id": sub["row_id"], "control_success": 0.5}).to_csv(
        f"{pkg}/data/sample_submission.csv", index=False, encoding="utf-8-sig")
code = ("import runpy,io,contextlib,time;t0=time.time()\n"
        "with contextlib.redirect_stdout(io.StringIO()) as f:\n"
        "    runpy.run_path('script.py', run_name='__main__')\n"
        "print(f'{time.time()-t0:.1f}')\n")
seq = ["submit_44", "submit_53", "submit_44", "submit_53"]
ts = {p: [] for p in set(seq)}
for pkg in seq:
    r = subprocess.run([sys.executable, "-c", code], cwd=os.path.join(ROOT, pkg),
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=ENV)
    t = float(r.stdout.strip().splitlines()[0])
    ts[pkg].append(t)
    print(f"  {pkg}  {t:.1f}s", flush=True)
t44 = min(ts["submit_44"]); t53 = min(ts["submit_53"])
print(f"\n  최소값 기준  44={t44:.1f}s  53={t53:.1f}s  "
      f"서버추정 {t53*261/t44/60:.2f}분  {'통과' if t53*261/t44 <= 480 else '**초과**'}")
t44m = np.mean(ts["submit_44"]); t53m = np.mean(ts["submit_53"])
print(f"  평균값 기준  44={t44m:.1f}s  53={t53m:.1f}s  서버추정 {t53m*261/t44m/60:.2f}분")
for p in set(seq):
    shutil.rmtree(f"{p}/data", ignore_errors=True)
    shutil.rmtree(f"{p}/output", ignore_errors=True)
    shutil.rmtree(f"{p}/__pycache__", ignore_errors=True)
