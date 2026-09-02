# -*- coding: utf-8 -*-
"""제출본 submit_ens 를 실전 규모로 검증한다.

세 가지를 본다
    1. 우리 build_features(제출 script.py) 와 features44.build(학습에 쓴 것) 가
       값까지 같은가.  다르면 flatMLP 가 학습과 다른 입력을 받는다.
    2. 253,507 행에서 script.py 전체 실행 시간.  서버 제한은 10분.
    3. 행 순서를 섞어도 같은 예측이 나오는가 (행 독립성 규칙).
"""
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# 서버에서도 로컬에서도 돌 수 있게 경로는 환경변수로 받는다.
#   ENS_ROOT   train.csv / test.csv 가 들어있는 data 의 부모
#   ENS_SUB    제출본 사본을 풀어놓고 실행할 작업 폴더
W = os.environ.get("ENS_ROOT", "/workspace/aimers")
SUB = os.environ.get("ENS_SUB", os.path.join(W, "submit_ens"))
sys.path.insert(0, SUB)
sys.path.insert(0, os.environ.get("ENS_CODE", os.path.join(W, "colab")))


# 학습 경로(train mode)와 추론 경로(inference mode)가 원래 다르게 나오는 열.
# 히스토리 테이블에서 나오는 계열이다. 학습 때는 프레임 안에서 시즌 진행에 따라
# 누적되고, 추론 때는 얼려둔 테이블 값을 그대로 쓴다. 설계상 그렇게 돼 있고
# CatBoost 도 같은 비대칭 위에서 1021 을 받았다. 여기서 확인할 것은
# "그 계열 밖에서는 차이가 없는가" 다.
HISTORY_FAMILY = {
    "p_is_succ", "pn_cur", "b_is_succ",          # 인시즌 누적
    "lg_cm_eff", "cm_rel", "p_adj_cm",           # 카운트 매치업
    "plat_dev",                                  # 플래툰
}


def check_features():
    """제출 build_features 와 학습에 쓴 features44 를 맞대본다.

    float32 저장 때문에 상대오차 1e-6 수준은 정상이다. 그보다 크게 벌어지는
    열이 HISTORY_FAMILY 밖에 있으면 그때가 진짜 문제다.
    """
    import features44 as F
    import script as S

    d = F.build(os.path.join(W, "data"), VS=2025)
    A, names = d["X44"], list(d["F44"])       # X44 는 numpy 배열, 열 이름은 F44
    tr = pd.read_csv(os.path.join(W, "data", "train.csv"), encoding="utf-8-sig")
    hist, _z, _s = S.load_model()
    B = S.build_features(tr, hist)

    print(f"  features44 {A.shape} {A.dtype}   제출 build_features {B.shape}")
    if names != list(B.columns):
        print("  !! 열 이름/순서 불일치")
        return
    unexpected = []
    for k, c in enumerate(names):
        u = np.asarray(A[:, k], dtype=np.float64)
        v = B[c].to_numpy(dtype=np.float64)
        nanmis = int((np.isnan(u) != np.isnan(v)).sum())
        m = ~(np.isnan(u) | np.isnan(v))
        scale = max(np.abs(u[m]).max(), 1.0) if m.any() else 1.0
        rel = (np.abs(u[m] - v[m]).max() / scale) if m.any() else 0.0
        if c in HISTORY_FAMILY:
            print(f"  (예상) {c:12s} 상대차 {rel:.2e}")
        elif rel > 1e-5 or nanmis:
            unexpected.append((c, rel, nanmis))
    if unexpected:
        for c, rel, n in unexpected:
            print(f"  !! 예상 밖 {c:24s} 상대차 {rel:.2e}  NaN불일치 {n}")
    else:
        print("  히스토리 계열 밖 37열은 float32 오차 이내로 일치")


def make_big_test():
    """train.csv 로 253,507 행짜리 가짜 test.csv 를 만든다. 정답 열은 뺀다."""
    tr = pd.read_csv(os.path.join(W, "data", "train.csv"), encoding="utf-8-sig")
    te = pd.read_csv(os.path.join(W, "data", "test.csv"), encoding="utf-8-sig")
    keep = [c for c in te.columns]
    big = tr.sample(n=253507, random_state=0, replace=len(tr) < 253507).copy()
    big = big.reindex(columns=keep)
    big["row_id"] = [f"TEST_{i:06d}" for i in range(len(big))]
    os.makedirs(os.path.join(SUB, "data"), exist_ok=True)
    big.to_csv(os.path.join(SUB, "data", "test.csv"), index=False)
    pd.DataFrame({"row_id": big["row_id"], "control_success": 0.5}).to_csv(
        os.path.join(SUB, "data", "sample_submission.csv"), index=False)
    print(f"  가짜 test {big.shape}  열 {list(big.columns)[:5]} ...")
    return big


def run_timed(tag):
    t = time.time()
    r = subprocess.run([sys.executable, "script.py"], cwd=SUB,
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    el = time.time() - t
    print(r.stdout[-1400:])
    if r.returncode:
        print(r.stderr[-3000:])
        raise SystemExit(1)
    print(f"  [{tag}] {el/60:.2f} 분")
    return pd.read_csv(os.path.join(SUB, "output", "submission.csv"))


if __name__ == "__main__":
    # 1.47M 행 피처 재생성은 몇 분 걸린다. 한 번 확인했으면 건너뛴다.
    if os.environ.get("SKIP_FEATCHECK") != "1":
        print("== 1. 피처 값 대조 ==", flush=True)
        check_features()

    print("\n== 2. 253,507 행 실행 시간 ==", flush=True)
    big = make_big_test()
    s1 = run_timed("정순")

    print("\n== 3. 행 순서 뒤섞기 ==", flush=True)
    sh = big.sample(frac=1.0, random_state=7)
    sh.to_csv(os.path.join(SUB, "data", "test.csv"), index=False)
    pd.DataFrame({"row_id": sh["row_id"], "control_success": 0.5}).to_csv(
        os.path.join(SUB, "data", "sample_submission.csv"), index=False)
    s2 = run_timed("역순")
    a = s1.set_index("row_id").iloc[:, 0]
    b = s2.set_index("row_id").iloc[:, 0].reindex(a.index)
    print(f"  최대차 {np.abs(a.to_numpy() - b.to_numpy()).max():.3e}"
          f"  (0 이어야 행 독립)")
