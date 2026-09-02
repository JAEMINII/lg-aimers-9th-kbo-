# -*- coding: utf-8 -*-
"""제출 직전 시프트 재보정. 로컬 CPU 로 돈다 (GPU 불필요).

왜 필요한가
    로짓 시프트는 모델에 딸린 값이지 상수가 아니다. submit_19 는 submit_18 의
    -0.005 를 그대로 물려받았는데, 같은 60,000행에서 예측 평균이 0.4843 ->
    0.4893 으로 +0.0050 올라 있었다. 손실 근사로 10~20점이다. 28점 낙폭의 절반이다.

무엇을 하나
    기준 제출본(점수를 아는 것)과 후보 제출본을 **같은 입력**에 돌려
    예측 평균 차이 D 를 잰다. 최적 시프트가 c* = -4D 이므로 후보의 시프트는
        새 시프트 = 기준 시프트 - 4 x (후보평균 - 기준평균)
    가 된다. 2025 의 실제 성공률은 못 보지만 '기준 대비 얼마나 옮길지' 는 잰다.

    입력은 train.csv 의 2024 시즌 전체에서 뽑는다. 253,507행이라 배치와 크기가
    같고 1군/퓨처스 비율(11.8%)도 유지된다. 라벨은 안 쓴다 — 두 모델 다
    2024 를 학습해서 채점하면 누출이다. 예측 **평균 차이**만 쓴다.

사용법
    python colab/shift_recal.py <기준.zip> <후보.zip> [행수]
예)
    python colab/shift_recal.py jaemin_1046_except_MLP.zip submit_jaemin_20.zip 60000
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

REF_OPT = -0.0070      # 기준 계열의 최적 시프트. 모르면 None 으로 두면 사용값을 쓴다.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "open (1)", "data")
PY = sys.executable


def stage(zpath, work, test_df):
    d = os.path.join(work, os.path.basename(zpath).replace(".zip", ""))
    os.makedirs(d, exist_ok=True)
    zipfile.ZipFile(zpath).extractall(d)
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    test_df.to_csv(os.path.join(d, "data", "test.csv"), index=False,
                   encoding="utf-8")
    pd.DataFrame({"row_id": test_df.row_id,
                  "control_success": 0.5}).to_csv(
        os.path.join(d, "data", "sample_submission.csv"), index=False,
        encoding="utf-8")
    return d


def cur_shift(d):
    src = open(os.path.join(d, "script.py"), encoding="utf-8").read()
    m = re.search(r"^CALIB_LOGIT_SHIFT\s*=\s*(-?[\d.eE+-]+)", src, re.M)
    return float(m.group(1)) if m else float("nan")


def run(d):
    r = subprocess.run([PY, "script.py"], cwd=d, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"{d} 실행 실패 rc={r.returncode}")
    out = os.path.join(d, "output", "submission.csv")
    return pd.read_csv(out), r.stdout


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    ref_z, cand_z = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 60000

    tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
    te = pd.read_csv(os.path.join(DATA, "test.csv"), encoding="utf-8-sig")
    s24 = tr[tr.season == 2024]
    rng = np.random.default_rng(0)
    idx = np.sort(rng.permutation(len(s24))[:min(n, len(s24))])
    big = s24.iloc[idx].reset_index(drop=True)
    isf = (big.game_type == "F").to_numpy()
    tdf = big.reindex(columns=list(te.columns))
    print(f"  입력 {len(tdf):,}행   퓨처스 {isf.mean()*100:.1f}%")

    work = tempfile.mkdtemp(prefix="shiftrecal_")
    try:
        res = {}
        for tag, z in (("기준", ref_z), ("후보", cand_z)):
            d = stage(os.path.join(ROOT, z), work, tdf)
            sub, _ = run(d)
            p = sub.set_index("row_id").loc[tdf.row_id, "control_success"].to_numpy()
            res[tag] = (p, cur_shift(d))
            print(f"  {tag} {z}  시프트 {res[tag][1]:+.4f}  "
                  f"평균 {p.mean():.4f}  SD {p.std():.5f}")

        # 앵커는 기준 제출본이 **쓴 값**이 아니라 그 계열의 **최적값**이다.
        # submit_20 은 -0.0040 을 썼지만 리더보드 2점(1057@-0.004, 1050@-0.024)
        # 에서 역산한 최적은 -0.0070 이다. 쓴 값을 앵커로 삼으면 그 오차가
        # 그대로 후보에 전파된다.
        pr, sr_used = res["기준"]
        sr = REF_OPT if REF_OPT is not None else sr_used
        pc, sc = res["후보"]
        D = float(pc.mean() - pr.mean())
        new = sr - 4.0 * D
        print(f"\n  예측평균 차이 D = {D:+.5f}")
        print(f"    1군    {pc[~isf].mean()-pr[~isf].mean():+.5f}")
        print(f"    퓨처스 {pc[isf].mean()-pr[isf].mean():+.5f}")
        print(f"  앵커(기준 계열 최적) {sr:+.4f}  [기준이 실제로 쓴 값 {sr_used:+.4f}]")
        print(f"  ->  후보 권장 시프트 {new:+.4f}")
        cost = 25000.0 * (sc - new) ** 2
        print(f"  후보를 {sc:+.4f} 로 그대로 두면 최적 대비 약 {cost:.0f}점 손해")
        print(f"  예측 SD  기준 {pr.std():.5f}  후보 {pc.std():.5f}  "
              f"비율 {pc.std()/pr.std():.4f}")
        print("\n  주의: 2024 는 두 모델 모두의 학습 구간이다. 점수는 못 읽고")
        print("  '기준 대비 평균이 얼마나 움직였나' 만 읽는 값이다.")
    finally:
        shutil.rmtree(work, ignore_errors=True)
