# -*- coding: utf-8 -*-
"""61 검증: 58 대조 — 표본 동일성, 강제 stale(1군) 활성, 강제 cold 리그분기
(cold∩1군 은 58 과 동일해야 하고 cold∩퓨처스만 달라야 함), 시간, 규칙4."""
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
ROOT = os.getcwd()
DATA = "open (1)/data"
T58_SERVER_MIN = 6.54
cols = list(pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns)
tr = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig")
pool = tr[tr.season == 2024]
rng = np.random.default_rng(0)
pick = np.sort(rng.choice(len(pool), 20000, replace=False))
sub = pool.iloc[pick][cols].copy()
PKGS = ("submit_58", "submit_61")
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
a58 = np.array([P["submit_58"][r] for r in rid])
a61 = np.array([P["submit_61"][r] for r in rid])
est = T["submit_61"] / T["submit_58"] * T58_SERVER_MIN
print("  시간  61=%.1fs 58=%.1fs  ->  61 서버추정 %.2f분  %s" %
      (T["submit_61"], T["submit_58"], est,
       "통과" if est <= 8.0 else "**초과**"))
d0 = np.abs(a61 - a58).max()
print("  2024표본 58-61 최대차 %.2e %s (cold/stale 미활성)" %
      (d0, "통과" if d0 < 1e-9 else "**다름**"))

pid = sub["pitcher_id"].to_numpy()
bid = sub["batter_id"].to_numpy()
isf_s = sub["game_type"].astype(str).to_numpy() == "F"
wz61 = np.load("submit_61/model/warm_ids.npz", allow_pickle=False)
orig61 = {k: wz61[k] for k in wz61.files}
wz58 = np.load("submit_58/model/warm_ids.npz", allow_pickle=False)
orig58 = {k: wz58[k] for k in wz58.files}

# --- 강제 stale(1군): 리그/stale 집합만 2023 기준, 전역 warm 유지
tr3 = tr[tr.season <= 2023]
isf3 = tr3.game_type.astype(str).to_numpy() == "F"
pF = np.unique(tr3.pitcher_id.to_numpy()[isf3]).astype(np.int64)
pR = np.unique(tr3.pitcher_id.to_numpy()[~isf3]).astype(np.int64)
bF = np.unique(tr3.batter_id.to_numpy()[isf3]).astype(np.int64)
bR = np.unique(tr3.batter_id.to_numpy()[~isf3]).astype(np.int64)
last3 = tr3.groupby("pitcher_id")["season"].max()
st3 = np.array(sorted(int(k) for k, v in last3.items() if v <= 2022), np.int64)
np.savez_compressed("submit_61/model/warm_ids.npz",
                    pitcher_id=orig61["pitcher_id"],
                    batter_id=orig61["batter_id"],
                    pitcher_id_F=pF, pitcher_id_R=pR,
                    batter_id_F=bF, batter_id_R=bR, stale_pitcher_id=st3)
try:
    _, Pf = run("submit_61")
    af = np.array([Pf[r] for r in rid])
    wp = set(int(v) for v in orig61["pitcher_id"])
    wb = set(int(v) for v in orig61["batter_id"])
    cold = np.array([(int(p) not in wp) or (int(b) not in wb)
                     for p, b in zip(pid, bid)])
    spF, spR, sbF, sbR = set(pF), set(pR), set(bF), set(bR)
    dc = np.array([(int(p) not in (spF if f else spR))
                   or (int(b) not in (sbF if f else sbR))
                   for p, b, f in zip(pid, bid, isf_s)])
    stset = set(int(v) for v in st3)
    st = ~cold & ~dc & np.array([int(p) in stset for p in pid]) & ~isf_s
    changed = np.abs(af - a61) > 1e-12
    ok = (changed == st).all()
    print("  강제stale(1군)  기대 %d행 (%.3f)  변경 %d행  %s" %
          (st.sum(), st.mean(), changed.sum(),
           "일치 통과" if ok else "**불일치**"))
finally:
    np.savez_compressed("submit_61/model/warm_ids.npz", **orig61)

# --- 강제 cold: 전역 warm 을 2023 기준으로 축소 (stale 빔), 58/61 동시
wp3 = np.unique(tr3.pitcher_id.to_numpy()).astype(np.int64)
wb3 = np.unique(tr3.batter_id.to_numpy()).astype(np.int64)
np.savez_compressed("submit_58/model/warm_ids.npz",
                    pitcher_id=wp3, batter_id=wb3)
np.savez_compressed("submit_61/model/warm_ids.npz",
                    pitcher_id=wp3, batter_id=wb3,
                    pitcher_id_F=orig61["pitcher_id_F"],
                    pitcher_id_R=orig61["pitcher_id_R"],
                    batter_id_F=orig61["batter_id_F"],
                    batter_id_R=orig61["batter_id_R"],
                    stale_pitcher_id=np.array([], np.int64))
try:
    _, P58f = run("submit_58")
    _, P61f = run("submit_61")
    b58f = np.array([P58f[r] for r in rid])
    b61f = np.array([P61f[r] for r in rid])
    wp3s = set(int(v) for v in wp3)
    wb3s = set(int(v) for v in wb3)
    coldf = np.array([(int(p) not in wp3s) or (int(b) not in wb3s)
                      for p, b in zip(pid, bid)])
    cR = coldf & ~isf_s
    cF = coldf & isf_s
    dwarm = np.abs(b61f[~coldf] - b58f[~coldf]).max() if (~coldf).any() else 0
    dcr = np.abs(b61f[cR] - b58f[cR]).max() if cR.any() else 0
    ndf = int((np.abs(b61f[cF] - b58f[cF]) > 1e-12).sum()) if cF.any() else 0
    print("  강제cold  cold %.3f (1군 %d / 퓨처스 %d)" %
          (coldf.mean(), cR.sum(), cF.sum()))
    print("    warm행 58=61 최대차 %.2e %s" %
          (dwarm, "통과" if dwarm < 1e-9 else "**다름**"))
    print("    cold∩1군 58=61 최대차 %.2e %s (둘 다 msaif 라 동일해야)" %
          (dcr, "통과" if dcr < 1e-9 else "**다름**"))
    print("    cold∩퓨처스 상이 %d/%d %s (61 만 구MS 라 달라야)" %
          (ndf, cF.sum(), "통과" if ndf == cF.sum() else "**불일치**"))
finally:
    np.savez_compressed("submit_58/model/warm_ids.npz", **orig58)
    np.savez_compressed("submit_61/model/warm_ids.npz", **orig61)

# --- 규칙4: 순서 섞기
full = pd.read_csv("submit_61/data/test.csv", encoding="utf-8-sig")
per = rng.permutation(len(full))
full.iloc[per].to_csv("submit_61/data/test.csv", index=False,
                      encoding="utf-8-sig")
pd.DataFrame({"row_id": full["row_id"].to_numpy()[per],
              "control_success": 0.5}).to_csv(
    "submit_61/data/sample_submission.csv", index=False, encoding="utf-8-sig")
_, Pp = run("submit_61")
d2 = float(max(abs(P["submit_61"][r] - v) for r, v in Pp.items()))
print("  규칙4 순서 %.2e %s" % (d2, "통과" if d2 < 1e-6 else "**위반**"))
for p in PKGS:
    shutil.rmtree(f"{p}/data", ignore_errors=True)
    shutil.rmtree(f"{p}/output", ignore_errors=True)
    shutil.rmtree(f"{p}/__pycache__", ignore_errors=True)
print("검증 끝")
