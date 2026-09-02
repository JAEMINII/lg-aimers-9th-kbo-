import os, shutil, subprocess, sys
import numpy as np, pandas as pd
ENV = dict(os.environ, PYTHONIOENCODING="utf-8"); ROOT = os.getcwd()
DATA = "open (1)/data"
cols = list(pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns)
tr = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig"); pool = tr[tr.season == 2024]
rng = np.random.default_rng(0)
pick = np.sort(rng.choice(len(pool), 30000, replace=False))
sub = pool.iloc[pick][cols].copy()
y = pool.iloc[pick]["control_success"].to_numpy(np.float64)
os.makedirs("submit_35/data", exist_ok=True)
sub.to_csv("submit_35/data/test.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"row_id": sub["row_id"], "control_success": 0.5}).to_csv(
    "submit_35/data/sample_submission.csv", index=False, encoding="utf-8-sig")
subprocess.run([sys.executable, "-c",
    "import runpy,io,contextlib\nwith contextlib.redirect_stdout(io.StringIO()):\n"
    "    runpy.run_path('script.py', run_name='__main__')"],
    cwd=os.path.join(ROOT, "submit_35"), capture_output=True, text=True,
    encoding="utf-8", errors="replace", env=ENV)
p = pd.read_csv("submit_35/output/submission.csv")["control_success"].to_numpy()
V = y.mean()*(1-y.mean())
def lg(q):
    q = np.clip(q, 1e-6, 1-1e-6); return np.log(q/(1-q))
def sh(q, c): return 1/(1+np.exp(-(lg(q)+c)))
def sc(q): return 100000.0*(1-((q-y)**2).mean()/V)
CUR = -0.007
print(f"  현행 시프트 {CUR}   2024 표본 {len(y):,}행  실제 {y.mean():.5f}\n")
print(f"  {'총 시프트':>10s} {'예측평균':>10s} {'점수':>9s} {'현행대비':>9s}")
grid = [-0.002, -0.005, -0.007, -0.010, -0.012, -0.015, -0.020, -0.030, -0.040]
base = sc(p)
for tot in grid:
    q = sh(p, tot - CUR)
    m = ("   <- 현행" if abs(tot - CUR) < 1e-9 else "")
    print(f"  {tot:10.3f} {q.mean():10.5f} {sc(q):9.1f} {sc(q)-base:+9.1f}{m}")
xs = np.arange(-0.06, 0.01, 0.0005)
ss = np.array([sc(sh(p, x - CUR)) for x in xs])
b = xs[int(np.argmax(ss))]
i = int(np.argmax(ss))
curv = abs((ss[i+1] - 2*ss[i] + ss[i-1]) / 0.0005**2)
print(f"\n  최적 총 시프트 {b:+.4f}   최고점 {ss.max():.1f}   "
      f"현행 대비 {ss.max()-base:+.1f}")
print(f"  곡률 {curv:,.0f}  ->  0.01 벗어나면 {curv*0.0001/2:.1f}점, "
      f"0.033 벗어나면 {curv*0.033**2/2:.1f}점")
for d in ("data", "output", "__pycache__"):
    shutil.rmtree(f"submit_35/{d}", ignore_errors=True)
