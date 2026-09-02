# -*- coding: utf-8 -*-
"""submit_15(1046)에서 flatMLP 만 빼서 그 값어치를 리더보드로 직접 잰다.

왜 이 실험인가
    flatMLP 은 비중 0.30 인데 관문 단독 802.9 로 셋 중 가장 약하다
    (TabM 911, CatBoost 903). 그 비중을 정한 근거가 관문인데, 그 관문이
    flatMLP 을 과대평가한다는 걸 우리가 직접 측정했다.

        flatMLP 의 관문 기여        +45.5
        안 고른 후보 12개 평균 기여  +18.8
        -> 약 +22 가 '후보 10개 중 관문 최고를 고른' 선택 편향

    게다가 예측 평균이 실제보다 1.7%p 높아 미보정 상태다. 관문은 후보마다
    최적 시프트로 채점해 그걸 공짜로 고쳐 주는데 배치엔 그 서비스가 없다.

    그런데 flatMLP 을 뺀 판본을 리더보드에서 한 번도 안 재봤다.

기준을 submit_15 로 잡은 이유
    1046 으로 최고점이고, 그 뒤 관문 근거로 만든 두 제출은 1044 / 1041 로
    떨어졌다. 관문이 못 믿을 상태이니 실측이 가장 좋은 지점에서 한 가지만 바꾼다.

바꾸는 것
    비중    0.10 / 0.30 / 0.60  ->  0.14 / 0.00 / 0.86   (CB:TabM 비율 유지)
    시프트  -0.0145 -> -0.005

    시프트도 같이 바꾸는 이유: flatMLP 은 예측 평균을 올리는 모델이라 빼면
    혼합 평균이 0.4978 -> 0.4954 로 내려간다. 리더보드 두 점에서 역산한 관계
    (최적 시프트 = -4 x (예측평균 - 실제평균))로는 0.0024 x 4 = 0.0096 만큼
    덜 내려야 한다. 시프트를 그대로 두면 과보정이라 flatMLP 의 값어치가 아니라
    보정 오류를 재게 된다.
"""
import io
import os
import zipfile

ROOT = r"c:\Users\jaemin.DESKTOP-2B30D1D\Desktop\재민\재민공부\aimers_재민"
SRC = os.path.join(ROOT, "submit_jaemin_15.zip")
OUT = os.path.join(ROOT, "submit_jaemin_18.zip")
DROP = ("model/vs2025_seed42.npz", "model/vs2025_seed1.npz",
        "model/vs2025_seed777.npz", "model/prep_vs2025.npz",
        "model/vs2025_meta.json")

NEW_W = """# 3원 앙상블 비중.  최종 = W_CB x CatBoost + W_MLP x flatMLP + W_TABM x TabM
#
# 2026-08-23: flatMLP 을 뺀다. 그 값어치를 리더보드로 직접 재는 제출이다.
#
# flatMLP 은 관문 단독 802.9 로 셋 중 최약인데(TabM 911, CatBoost 903) 비중이
# 0.30 이었다. 그 비중의 근거가 관문인데 관문이 이 모델을 과대평가한다 —
#     flatMLP 관문 기여          +45.5
#     안 고른 후보 12개 평균 기여  +18.8
#     차이 약 +22 가 '후보 10개 중 관문 최고를 고른' 선택 편향
# 게다가 예측 평균이 실제보다 1.7%p 높다. 관문은 후보마다 최적 시프트로 채점해
# 그 편향을 공짜로 고쳐 주는데 배치엔 그 서비스가 없다.
#
# CatBoost 와 TabM 의 비율(1:6)은 그대로 두고 flatMLP 몫만 나눠 가진다.
W_CB, W_MLP, W_TABM = 0.14, 0.00, 0.86"""

NEW_SHIFT = """# 시프트도 같이 바꾼다. flatMLP 은 예측 평균을 올리는 모델이라 빼면 혼합 평균이
# 0.4978 -> 0.4954 로 내려간다(관문 기준). 리더보드 두 점에서 역산한 관계
#     최적 시프트 = -4 x (예측평균 - 실제평균)
# 로는 0.0024 x 4 = 0.0096 만큼 덜 내려야 한다. -0.0145 를 그대로 두면 과보정이라
# flatMLP 의 값어치가 아니라 보정 오류를 재게 된다.
CALIB_LOGIT_SHIFT = -0.005"""


if __name__ == "__main__":
    z = zipfile.ZipFile(SRC)
    s = z.read("script.py").decode("utf-8").replace("\r\n", "\n")

    i = s.index("# 3원 앙상블 비중.")
    j = s.index("W_CB, W_MLP, W_TABM = 0.10, 0.30, 0.60") + len(
        "W_CB, W_MLP, W_TABM = 0.10, 0.30, 0.60")
    s = s[:i] + NEW_W + s[j:]

    a = s.rindex("\n", 0, s.index("CALIB_LOGIT_SHIFT = -0.0145"))
    # 시프트 위 주석 블록 전체를 갈아끼운다
    while s[:a].rstrip().endswith(")") is False and s[a - 1:a] != "\n":
        a -= 1
    b = s.index("CALIB_LOGIT_SHIFT = -0.0145") + len("CALIB_LOGIT_SHIFT = -0.0145")
    head = s[:s.index("CALIB_LOGIT_SHIFT = -0.0145")]
    head = head[:head.rindex("# 시즌 드리프트 보정용 고정 로짓 시프트.")]
    s = head + NEW_SHIFT + s[b:]

    # flatMLP 로딩을 비중 0 이면 건너뛰게 한다
    old = '''        st = np.load(resolve("model/prep_vs2025.npz"), allow_pickle=False)'''
    new = '''        # 비중 0 이면 파일 자체를 안 읽는다. 제출본에서 뺐기 때문이다.
        if W_MLP == 0.0:
            p_mlp = np.zeros(len(X))
        else:
            st = np.load(resolve("model/prep_vs2025.npz"), allow_pickle=False)'''
    assert old in s
    s = s.replace(old, new)
    # 이어지는 flatMLP 블록을 else 안으로 들여쓴다
    k = s.index(new) + len(new)
    end = s.index('        # TabM', k)
    body = s[k:end]
    s = s[:k] + "\n".join(("    " + ln) if ln.strip() else ln
                          for ln in body.split("\n")) + s[end:]

    import ast
    ast.parse(s)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as w:
        w.writestr("script.py", s)
        for n in z.namelist():
            if n == "script.py" or n in DROP or n.endswith("/"):
                continue
            w.write_str = None
            w.writestr(n, z.read(n))
    print(f"{OUT}  {os.path.getsize(OUT)/1e6:.1f} MB")
    zz = zipfile.ZipFile(OUT)
    print("파일:", [n for n in zz.namelist()])
    for line in s.split("\n"):
        if line.startswith(("W_CB,", "CALIB_LOGIT_SHIFT")):
            print(" ", line)
