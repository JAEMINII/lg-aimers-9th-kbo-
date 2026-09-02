import os, shutil, subprocess, sys
import numpy as np, pandas as pd
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
ROOT = os.getcwd(); DATA = "open (1)/data"
cols = list(pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns)
tr = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig"); pool = tr[tr.season == 2024]
rng = np.random.default_rng(0)
pick = np.sort(rng.choice(len(pool), 30000, replace=False))
sub = pool.iloc[pick][cols].copy()
y = pool.iloc[pick]["control_success"].to_numpy(np.float64)
print(f"  2024 표본 {len(sub):,}행   실제 성공률 {y.mean():.5f}\n")
os.makedirs("submit_35/data", exist_ok=True)
sub.to_csv("submit_35/data/test.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"row_id": sub["row_id"], "control_success": 0.5}).to_csv(
    "submit_35/data/sample_submission.csv", index=False, encoding="utf-8-sig")
code = ("import runpy,io,contextlib\n"
        "with contextlib.redirect_stdout(io.StringIO()) as f:\n"
        "    runpy.run_path('script.py', run_name='__main__')\n"
        "print([l for l in f.getvalue().splitlines() if 'mean' in l])")
r = subprocess.run([sys.executable, "-c", code], cwd=os.path.join(ROOT, "submit_35"),
                   capture_output=True, text=True, encoding="utf-8",
                   errors="replace", env=ENV)
print("  submit_35 실제 실행:", r.stdout.strip()[:300])
p = pd.read_csv("submit_35/output/submission.csv")["control_success"].to_numpy()
V = y.mean()*(1-y.mean())
def sc(q): return 100000.0*(1-((q-y)**2).mean()/V)
def lg(q): 
    q=np.clip(q,1e-6,1-1e-6); return np.log(q/(1-q))
def sh(q,c): return 1/(1+np.exp(-(lg(q)+c)))
print(f"\n  제출본 예측 평균 {p.mean():.5f}   실제 {y.mean():.5f}   "
      f"차이 {p.mean()-y.mean():+.5f}")
print(f"\n  {'추가 시프트':>10s} {'평균':>10s} {'점수':>10s}")
for extra in (0.0, -0.01, -0.02, -0.033, -0.05):
    q = sh(p, extra)
    print(f"  {extra:10.3f} {q.mean():10.5f} {sc(q):10.1f}"
          + ("   <- 현행(-0.007)" if extra == 0 else
             "   <- -0.04 로 바꾸면" if extra == -0.033 else ""))
best = min(np.arange(-0.08, 0.02, 0.001), key=lambda c: ((sh(p,c)-y)**2).mean())
print(f"\n  이 표본에서의 최적 추가 시프트 {best:+.3f}  "
      f"(즉 총 시프트 {-0.007+best:+.3f})   점수 {sc(sh(p,best)):.1f}")
shutil.rmtree("submit_35/data", ignore_errors=True)
shutil.rmtree("submit_35/output", ignore_errors=True)
shutil.rmtree("submit_35/__pycache__", ignore_errors=True)
