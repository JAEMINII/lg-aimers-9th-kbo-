# -*- coding: utf-8 -*-
"""submit_27 = submit_20(1057) 에서 HistGB 를 빼고 CatBoost 0.3 / TabM 0.7.

바뀌는 것은 혼합 비중 하나뿐이다
    submit_20     0.10 CatBoost + 0.10 HistGB + 0.80 TabM
    submit_27     0.30 CatBoost + 0.00 HistGB + 0.70 TabM

    모델 파일도, TabM 경로도, 전처리도 전부 그대로다. 순수한 비중 A/B 다.

왜 관문으로 미리 안 고르나
    혼합 비중은 관문이 여섯 번 틀린 '조합' 부류다. 계열 간 순위를 못 매기고
    (CatBoost 관문 903 > TabM 868 인데 리더보드는 반대였다), 브랜치 구성에서도
    관문 -5.4 / 0-4 였던 게 리더보드 +3 이었다. 이 질문의 올바른 계기는
    리더보드 한 장이다.

HistGB 계산 자체를 건너뛴다
    비중만 0 으로 두면 결과는 같지만 추론 시간을 그대로 쓴다. submit_25 가
    9분 26초였고 예산이 10분이라 34초밖에 안 남았다. 트리 12,000개 순회를
    빼면 그 여유가 늘어난다. 모델 파일(hgb.npz)은 패키지에 남겨 둔다 —
    빼면 submit_20 과의 차이가 비중 하나가 아니게 되고, 되돌리기도 번거롭다.

시프트
    HistGB(평균 0.4303)를 빼고 CatBoost(0.4478) 비중을 올리면 혼합 평균이
    올라간다. 그대로 두면 절편이 어긋난다. shift_recal 로 submit_20 대비
    D 를 재고 -0.0070 - 4D 로 잡는다.
"""
import io
import os
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "submit_jaemin_20.zip")
OUT = os.path.join(ROOT, "submit_27")
W = (0.30, 0.00, 0.70)
SHIFT = -0.0070          # 자리표시. shift_recal 후 확정

OLD_HG = '''        gt = test["game_type"].astype(str).to_numpy()
        is_f_t = gt == "F"
        zh = np.load(resolve("model/hgb.npz"), allow_pickle=False)
        p_hgb = route_hgb(predict_hgb(X.values.astype(np.float64), zh), is_f_t)
        print(f" HistGB   mean={p_hgb.mean():.4f}  "
              f"상관={np.corrcoef(p_cb, p_hgb)[0, 1]:.4f}")'''

NEW_HG = '''        gt = test["game_type"].astype(str).to_numpy()
        is_f_t = gt == "F"
        if W_HGB > 0.0:
            zh = np.load(resolve("model/hgb.npz"), allow_pickle=False)
            p_hgb = route_hgb(predict_hgb(X.values.astype(np.float64), zh),
                              is_f_t)
            print(f" HistGB   mean={p_hgb.mean():.4f}  "
                  f"상관={np.corrcoef(p_cb, p_hgb)[0, 1]:.4f}")
        else:
            # 비중이 0 이면 트리 12,000개 순회를 통째로 건너뛴다.
            # 예산이 10분인데 submit_25 가 9분 26초였다.
            p_hgb = np.zeros_like(p_cb)
            print(" HistGB   건너뜀 (비중 0)")'''


def main():
    z = zipfile.ZipFile(SRC)
    src = z.read("script.py").decode("utf-8")

    old_w = "W_CB, W_HGB, W_TABM = 0.1, 0.1, 0.8"
    assert old_w in src, "비중 줄을 못 찾았다"
    src = src.replace(old_w, f"W_CB, W_HGB, W_TABM = {W[0]}, {W[1]}, {W[2]}")

    assert OLD_HG in src, "HistGB 블록을 못 찾았다"
    src = src.replace(OLD_HG, NEW_HG)

    old_s = "CALIB_LOGIT_SHIFT = -0.004"
    assert old_s in src, "시프트 줄을 못 찾았다"
    src = src.replace(old_s, f"CALIB_LOGIT_SHIFT = {SHIFT}")

    # 안전장치 — 나머지가 안 건드려졌는지
    assert "CALIB_T" in src, "온도 상수가 사라졌다"
    assert "fbregime_" in src, "퓨처스 경로가 사라졌다 (submit_20 그대로여야 한다)"

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "model"))
    io.open(os.path.join(OUT, "script.py"), "w", encoding="utf-8",
            newline="\n").write(src)
    keep = [n for n in z.namelist() if n != "script.py"]
    for n in keep:
        p = os.path.join(OUT, n)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        io.open(p, "wb").write(z.read(n))

    tot = sum(os.path.getsize(os.path.join(r, f))
              for r, _, fs in os.walk(OUT) for f in fs)
    print(f"  {OUT}  {tot/1e6:.1f}MB   비중 {W}   shift={SHIFT}")
    print(f"  submit_20 에서 그대로 가져온 파일 {len(keep)}개 "
          f"(script.py 만 새로 씀)")


if __name__ == "__main__":
    main()
