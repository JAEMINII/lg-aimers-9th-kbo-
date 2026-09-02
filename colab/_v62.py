# -*- coding: utf-8 -*-
"""62 검증: 61 대조 — 표본 동일성(dcp 는 train 표본에서 활성 불가),
강제 dcp 활성(리그집합 2023컷, stale 빔), 시간, 규칙4."""
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
ROOT = os.getcwd()
DATA = "open (1)/data"
T61_SERVER_MIN = 6.81
cols = list(pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns)
tr = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig")
pool = tr[tr.season == 2024]
rng = np.random.default_rng(0)
pick = np.sort(rng.choice(len(pool), 20000, replace=False))
sub = pool.iloc[pick][cols].copy()
PKGS = ("submit_61", "submit_62")
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
a61 = np.array([P["submit_61"][r] for r in rid])
a62 = np.array([P["submit_62"][r] for r in rid])
est = T["submit_62"] / T["submit_61"] * T61_SERVER_MIN
print("  시간  62=%.1fs 61=%.1fs  ->  62 서버추정 %.2f분  %s" %
      (T["submit_62"], T["submit_61"], est,
       "통과" if est <= 8.0 else "**초과**"))
d0 = np.abs(a62 - a61).max()
print("  2024표본 61-62 최대차 %.2e %s (dcp 활성 불가)" %
      (d0, "통과" if d0 < 1e-9 else "**다름**"))

pid = sub["pitcher_id"].to_numpy()
bid = sub["batter_id"].to_numpy()
isf_s = sub["game_type"].astype(str).to_numpy() == "F"
wz = np.load("submit_62/model/warm_ids.npz", allow_pickle=False)
orig = {k: wz[k] for k in wz.files}
tr3 = tr[tr.season <= 2023]
isf3 = tr3.game_type.astype(str).to_numpy() == "F"
pF = np.unique(tr3.pitcher_id.to_numpy()[isf3]).astype(np.int64)
pR = np.unique(tr3.pitcher_id.to_numpy()[~isf3]).astype(np.int64)
bF = np.unique(tr3.batter_id.to_numpy()[isf3]).astype(np.int64)
bR = np.unique(tr3.batter_id.to_numpy()[~isf3]).astype(np.int64)
np.savez_compressed("submit_62/model/warm_ids.npz",
                    pitcher_id=orig["pitcher_id"], batter_id=orig["batter_id"],
                    pitcher_id_F=pF, pitcher_id_R=pR,
                    batter_id_F=bF, batter_id_R=bR,
                    stale_pitcher_id=np.array([], np.int64))
try:
    _, Pf = run("submit_62")
    af = np.array([Pf[r] for r in rid])
    wp = set(int(v) for v in orig["pitcher_id"])
    wb = set(int(v) for v in orig["batter_id"])
    cold = np.array([(int(p) not in wp) or (int(b) not in wb)
                     for p, b in zip(pid, bid)])
    spF, spR = set(pF), set(pR)
    dcp = ~cold & np.array([int(p) not in (spF if f else spR)
                            for p, f in zip(pid, isf_s)]) & ~isf_s
    changed = np.abs(af - a62) > 1e-12
    ok = (changed == dcp).all()
    print("  강제dcp  기대 %d행 (%.4f)  변경 %d행  %s" %
          (dcp.sum(), dcp.mean(), changed.sum(),
           "일치 통과" if ok else "**불일치**"))
finally:
    np.savez_compressed("submit_62/model/warm_ids.npz", **orig)

full = pd.read_csv("submit_62/data/test.csv", encoding="utf-8-sig")
per = rng.permutation(len(full))
full.iloc[per].to_csv("submit_62/data/test.csv", index=False,
                      encoding="utf-8-sig")
pd.DataFrame({"row_id": full["row_id"].to_numpy()[per],
              "control_success": 0.5}).to_csv(
    "submit_62/data/sample_submission.csv", index=False, encoding="utf-8-sig")
_, Pp = run("submit_62")
d2 = float(max(abs(P["submit_62"][r] - v) for r, v in Pp.items()))
print("  규칙4 순서 %.2e %s" % (d2, "통과" if d2 < 1e-6 else "**위반**"))
for p in PKGS:
    shutil.rmtree(f"{p}/data", ignore_errors=True)
    shutil.rmtree(f"{p}/output", ignore_errors=True)
    shutil.rmtree(f"{p}/__pycache__", ignore_errors=True)
print("검증 끝")
