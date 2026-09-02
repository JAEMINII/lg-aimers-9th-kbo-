# -*- coding: utf-8 -*-
"""submit_27 과 submit_29 를 같은 행에서 돌려 차이를 해부한다.

왜
    관문 시드곡선이 k=1 -> k=8 에서 +7.9 였는데 리더보드는 1058 -> 1056 이었다.
    내가 '분산 축소 축이라 관문이 그대로 전이된다' 고 했는데 틀렸다.

    가장 먼저 의심할 것은 **시프트**다. 이전 제출은 전부 shift_recal 로
    재보정했는데 submit_29 는 submit_27 의 -0.0070 을 그대로 물려받았다.
    8시드 평균은 예측 평균이 움직일 수 있고 곡률이 25000 이라 0.01 어긋나면
    2.5점이 깎인다.

무엇을 보나
    같은 2024 행 60,000개에 두 패키지를 돌려서
      ① 최종 예측 평균 차이 D  -> 필요한 시프트 보정은 -4D
      ② TabM 단계 평균 차이    -> 시드 평균이 예측을 어느 쪽으로 미는지
      ③ 예측 표준편차 비율     -> 분산이 실제로 줄었는지
      ④ 실제 라벨로 채점        -> 2024 는 학습 구간이라 절대값은 못 읽지만
                                 같은 조건에서의 상대 비교는 된다
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = "/workspace/aimers"
N = 60000


def run(pkg):
    d = os.path.join(ROOT, pkg)
    os.makedirs(os.path.join(d, "data"), exist_ok=True)
    tr = pd.read_csv(f"{ROOT}/data/train.csv", encoding="utf-8-sig")
    tr = tr[tr.season == 2024].head(N).reset_index(drop=True)
    y = tr["control_success"].to_numpy(np.float64)
    tr.drop(columns=["control_success"]).to_csv(
        os.path.join(d, "data", "test.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame({"row_id": tr.row_id, "control_success": 0.5}).to_csv(
        os.path.join(d, "data", "sample_submission.csv"), index=False)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    out = subprocess.run([sys.executable, "-u", "script.py"], cwd=d, env=env,
                         capture_output=True, text=True)
    tail = [l for l in out.stdout.splitlines()
            if "mean=" in l or "시드" in l or "GPU=" in l]
    sub = pd.read_csv(os.path.join(d, "output", "submission.csv"))
    col = [c for c in sub.columns if c != "row_id"][0]
    return sub[col].to_numpy(np.float64), y, tail


def score(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - np.mean((p - y) ** 2) / (r * (1 - r)))


def best_shift(p, y):
    from scipy.optimize import minimize_scalar
    lg = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    f = lambda c: -score(1 / (1 + np.exp(-(lg + c))), y)      # noqa: E731
    r = minimize_scalar(f, bounds=(-0.5, 0.5), method="bounded")
    return -r.fun, r.x


if __name__ == "__main__":
    p27, y, t27 = run("s30cmp")
    p29, _, t29 = run("s33cmp")
    print("\n  submit_30:", " | ".join(t27))
    print("  submit_33:", " | ".join(t29))

    D = float(p29.mean() - p27.mean())
    print(f"\n  {'':14s} {'평균':>9s} {'표준편차':>9s} {'현시프트 점수':>12s} "
          f"{'최적시프트':>10s} {'최적 점수':>10s}")
    for nm, p in (("submit_30", p27), ("submit_33", p29)):
        s0 = score(p, y)
        sb, cb = best_shift(p, y)
        print(f"  {nm:14s} {p.mean():9.5f} {p.std():9.5f} {s0:12.1f} "
              f"{cb:+10.4f} {sb:10.1f}")

    print(f"\n  예측평균 차이 D = {D:+.6f}")
    print(f"  -> 필요한 시프트 보정 -4D = {-4*D:+.5f}")
    print(f"  -> submit_33 권장 시프트 = {-0.0070 - 4*D:+.5f}")
    print(f"  시프트를 안 고쳐서 잃은 점수 (곡률 25000) "
          f"= {25000*(4*D)**2:.2f}")
    print(f"\n  예측 표준편차 비율 29/27 = {p29.std()/p27.std():.5f}")
    print(f"  두 예측의 상관 = {np.corrcoef(p27, p29)[0,1]:.6f}")
    print(f"  행별 차이 |29-27| 평균 {np.abs(p29-p27).mean():.5f}  "
          f"최대 {np.abs(p29-p27).max():.5f}")
    print("\n  2024 는 두 모델 모두의 학습 구간이라 절대 점수는 못 읽는다.")
    print("  평균 이동과 분산 변화만 읽는다.")
