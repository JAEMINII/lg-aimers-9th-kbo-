import os, shutil, subprocess, sys, zipfile
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
import numpy as np, pandas as pd
ROOT = os.getcwd(); DATA = "open (1)/data"
cols = list(pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns)
tr = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig"); pool = tr[tr.season == 2024]
rng = np.random.default_rng(0)
pick = np.sort(rng.choice(len(pool), 20000, replace=False))
sub = pool.iloc[pick][cols].copy()
for pkg in ("submit_35", "submit_36"):
    os.makedirs(f"{pkg}/data", exist_ok=True)
    sub.to_csv(f"{pkg}/data/test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"row_id": sub["row_id"], "control_success": 0.5}).to_csv(
        f"{pkg}/data/sample_submission.csv", index=False, encoding="utf-8-sig")
code = ("import runpy,io,contextlib,time;t0=time.time()\n"
        "with contextlib.redirect_stdout(io.StringIO()) as f:\n"
        "    runpy.run_path('script.py', run_name='__main__')\n"
        "print(f'{time.time()-t0:.1f}')\n"
        "print([l for l in f.getvalue().splitlines() if 'TabM' in l or 'calibrated' in l])")
P = {}
for pkg in ("submit_35", "submit_36"):
    r = subprocess.run([sys.executable, "-c", code], cwd=os.path.join(ROOT, pkg),
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV)
    if r.returncode:
        print(f"  {pkg} 실패:\n{r.stderr[-900:]}"); sys.exit(1)
    lines = r.stdout.strip().splitlines()
    print(f"  {pkg}  {lines[0]}s   {lines[-1]}")
    P[pkg] = pd.read_csv(f"{pkg}/output/submission.csv")["control_success"].to_numpy()
isf = sub["game_type"].astype(str).to_numpy() == "F"
print(f"\n  35 vs 36   전체상관 {np.corrcoef(P['submit_35'], P['submit_36'])[0,1]:.5f}")
print(f"    1군    상관 {np.corrcoef(P['submit_35'][~isf], P['submit_36'][~isf])[0,1]:.5f}  "
      f"최대차 {np.abs(P['submit_35'][~isf]-P['submit_36'][~isf]).max():.5f}")
print(f"    퓨처스  최대차 {np.abs(P['submit_35'][isf]-P['submit_36'][isf]).max():.2e}  "
      f"(0 이어야 맞다 — 퓨처스 경로는 안 바꿨다)")
# 규칙 4
full = pd.read_csv("submit_36/data/test.csv", encoding="utf-8-sig")
base = dict(zip(pd.read_csv("submit_36/output/submission.csv")["row_id"],
                P["submit_36"]))
per = rng.permutation(len(full))
full.iloc[per].to_csv("submit_36/data/test.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"row_id": full["row_id"].to_numpy()[per],
              "control_success": 0.5}).to_csv(
    "submit_36/data/sample_submission.csv", index=False, encoding="utf-8-sig")
subprocess.run([sys.executable, "-c", code], cwd=os.path.join(ROOT, "submit_36"),
               capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV)
g = pd.read_csv("submit_36/output/submission.csv")
d = float(max(abs(base[r] - v) for r, v in zip(g["row_id"], g["control_success"])))
print(f"\n  규칙4 순서 섞기 최대차 {d:.3e}  {'통과' if d < 1e-9 else '**위반**'}")
for p in ("submit_35", "submit_36"):
    shutil.rmtree(f"{p}/data", ignore_errors=True)
    shutil.rmtree(f"{p}/output", ignore_errors=True)
    shutil.rmtree(f"{p}/__pycache__", ignore_errors=True)
SRC, OUT = "submit_36", "submit_jaemin_36.zip"
files = ["preprocess.py", "requirements.txt", "script.py"] + \
        [f"model/{f}" for f in sorted(os.listdir(f"{SRC}/model"))]
if os.path.exists(OUT): os.remove(OUT)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for f in files: z.write(os.path.join(SRC, f), f)
z = zipfile.ZipFile(OUT)
print(f"\n  {OUT}  파일 {len(z.infolist())}개  "
      f"압축 {os.path.getsize(OUT)/1e6:.1f}MB  무결성 {z.testzip() or '이상 없음'}")
