# -*- coding: utf-8 -*-
"""53 검증: 44앵커 시간환산, 52 대조(퓨처스 동일성), 규칙4, 수준, 상관."""
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
y24 = pool.iloc[pick]["control_success"].to_numpy(float)
PKGS = ("submit_44", "submit_52", "submit_53")
for pkg in PKGS:
    os.makedirs(f"{pkg}/data", exist_ok=True)
    sub.to_csv(f"{pkg}/data/test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"row_id": sub["row_id"], "control_success": 0.5}).to_csv(
        f"{pkg}/data/sample_submission.csv", index=False, encoding="utf-8-sig")
code = ("import runpy,io,contextlib,time;t0=time.time()\n"
        "with contextlib.redirect_stdout(io.StringIO()) as f:\n"
        "    runpy.run_path('script.py', run_name='__main__')\n"
        "print(f'{time.time()-t0:.1f}')\n")
T, P = {}, {}
for pkg in PKGS:
    r = subprocess.run([sys.executable, "-c", code], cwd=os.path.join(ROOT, pkg),
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=ENV)
    if r.returncode:
        print(f"  {pkg} 실패:\n{r.stderr[-1500:]}"); sys.exit(1)
    T[pkg] = float(r.stdout.strip().splitlines()[0])
    g = pd.read_csv(f"{pkg}/output/submission.csv")
    P[pkg] = dict(zip(g["row_id"], g["control_success"]))
    print(f"  {pkg}  {T[pkg]:.1f}s")
rid = sub["row_id"].tolist()
a44 = np.array([P["submit_44"][r] for r in rid])
a52 = np.array([P["submit_52"][r] for r in rid])
a53 = np.array([P["submit_53"][r] for r in rid])
est = T["submit_53"] * 261.0 / T["submit_44"]
print(f"\n  시간  53={T['submit_53']:.1f}s 44={T['submit_44']:.1f}s(눈금 261s)"
      f"  ->  53 서버추정 {est/60:.1f}분  {'통과' if est <= 480 else '**초과**'}")
isf = sub["game_type"].astype(str).to_numpy() == "F"
df = np.abs(a53[isf] - a52[isf]).max()
print(f"  퓨처스 52↔53 최대차 {df:.2e}  {'(동일해야 함) 통과' if df < 1e-9 else '**다름**'}")
print(f"  상관  44↔53 {np.corrcoef(a44, a53)[0,1]:.5f}   52↔53 {np.corrcoef(a52, a53)[0,1]:.5f}"
      f"   1군 52↔53 {np.corrcoef(a52[~isf], a53[~isf])[0,1]:.5f}")
def zshift(m_from, m_to):
    l = lambda p: np.log(p/(1-p))
    return l(m_to) - l(m_from)
print(f"  수준  실제 {y24.mean():.5f}  44 {a44.mean():.5f}  52 {a52.mean():.5f}  53 {a53.mean():.5f}"
      f"  Δz(53-실제) {zshift(y24.mean(), a53.mean()):+.5f}")
# 규칙4: 순서 섞기 + 부분집합
full = pd.read_csv("submit_53/data/test.csv", encoding="utf-8-sig")
per = rng.permutation(len(full))
full.iloc[per].to_csv("submit_53/data/test.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"row_id": full["row_id"].to_numpy()[per], "control_success": 0.5}).to_csv(
    "submit_53/data/sample_submission.csv", index=False, encoding="utf-8-sig")
subprocess.run([sys.executable, "-c", code], cwd=os.path.join(ROOT, "submit_53"),
               capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV)
g = pd.read_csv("submit_53/output/submission.csv")
d1 = float(max(abs(P["submit_53"][r] - v) for r, v in zip(g["row_id"], g["control_success"])))
half = full.iloc[np.sort(rng.choice(len(full), 7000, replace=False))]
half.to_csv("submit_53/data/test.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"row_id": half["row_id"], "control_success": 0.5}).to_csv(
    "submit_53/data/sample_submission.csv", index=False, encoding="utf-8-sig")
subprocess.run([sys.executable, "-c", code], cwd=os.path.join(ROOT, "submit_53"),
               capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV)
g = pd.read_csv("submit_53/output/submission.csv")
d2 = float(max(abs(P["submit_53"][r] - v) for r, v in zip(g["row_id"], g["control_success"])))
print(f"  규칙4  순서 {d1:.2e} {'통과' if d1 < 1e-9 else '**위반**'}   "
      f"부분 {d2:.2e} {'통과' if d2 < 1e-9 else '**위반**'}")
for p in PKGS:
    shutil.rmtree(f"{p}/data", ignore_errors=True)
    shutil.rmtree(f"{p}/output", ignore_errors=True)
    shutil.rmtree(f"{p}/__pycache__", ignore_errors=True)
