# -*- coding: utf-8 -*-
"""제출본이 대회 규칙 4(행 독립 추론)를 지키는지 실증으로 확인한다.

규칙 4
    평가 데이터의 각 행은 독립적인 예측 대상이다. 다른 행이나 전체 평가 데이터의
    분포를 이용해 특정 행의 예측값을 보정·생성하면 안 된다.

코드를 읽는 것만으로는 부족하다. 두 가지를 실제로 돌려서 비교한다.

    순서 검사   행 순서를 섞어도 같은 row_id 는 같은 값이 나오는가
                -> 순서 의존을 잡는다
    부분 검사   전체의 20% 만 넣어도 그 행들의 값이 그대로인가
                -> 전체 분포 의존을 잡는다. 순서 검사로는 안 잡힌다.
                   (평균 같은 걸 썼다면 순서를 바꿔도 값이 안 변한다)

둘 다 차이 0 이어야 통과다.
"""
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

SUB = os.environ["ENS_SUB"]
ROOT = os.environ["ENS_ROOT"]
N = int(os.environ.get("AUDIT_N", "60000"))
TOL = 1e-6      # 부동소수점 반올림은 통과, 정보 유출은 걸리는 선


def write_test(df):
    df.to_csv(os.path.join(SUB, "data", "test.csv"), index=False)
    pd.DataFrame({"row_id": df["row_id"], "control_success": 0.5}).to_csv(
        os.path.join(SUB, "data", "sample_submission.csv"), index=False)


def run(tag):
    t = time.time()
    r = subprocess.run([sys.executable, "script.py"], cwd=SUB, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(1)
    out = pd.read_csv(os.path.join(SUB, "output", "submission.csv"))
    print(f"  [{tag}] {len(out):,}행  {time.time()-t:.0f}초")
    return out.set_index("row_id").iloc[:, 0]


if __name__ == "__main__":
    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), encoding="utf-8-sig")
    te = pd.read_csv(os.path.join(ROOT, "data", "test.csv"), encoding="utf-8-sig")
    big = tr.sample(n=N, random_state=0).reindex(columns=list(te.columns))
    big["row_id"] = [f"TEST_{i:06d}" for i in range(len(big))]

    print(f"== 기준 ({N:,}행) ==", flush=True)
    write_test(big)
    base = run("기준")

    print("\n== 순서 검사: 행을 섞어서 다시 ==", flush=True)
    write_test(big.sample(frac=1.0, random_state=7))
    shuf = run("섞음")
    d1 = np.abs(base.to_numpy() - shuf.reindex(base.index).to_numpy()).max()
    print(f"  최대차 {d1:.3e}   {'통과' if d1 < TOL else '위반'}")

    print("\n== 부분 검사: 20% 만 넣어서 다시 ==", flush=True)
    part = big.sample(frac=0.2, random_state=11)
    write_test(part)
    sm = run("부분")
    idx = sm.index
    d2 = np.abs(base.reindex(idx).to_numpy() - sm.to_numpy()).max()
    print(f"  최대차 {d2:.3e}   {'통과' if d2 < TOL else '위반'}")

    print(f"\n규칙 4 판정: {'통과' if max(d1, d2) < TOL else '위반'}   (판정선 {TOL:.0e})")
