# -*- coding: utf-8 -*-
"""mlp2024.pkl -> mlp.npz  (순수 numpy 추론용)

채점 환경에 torch/pytabkit 이 없을 수 있어 가중치와 전처리 파라미터를 배열로 뽑고
numpy 로 재현한다. CatBoost 트리를 같은 방식으로 뽑아 24.6만 행 오차 2.2e-16 을
확인한 전례가 있다 (submit_jaemin_6/export_trees.py).

pytabkit 파이프라인 (소스 + 실측으로 확인)
  ToDictDatasetConverter   원값 -> OrdinalEncoder(+1). 미지값 0
    x_cont 44열 / x_cat 16열 로 쪼개진다. 이때 '수치가 앞, 범주가 뒤' 로 재배치된다
  tfm = SequentialLayer([SklearnTransformLayer(TabrQuantileTransformer)])
    x_cont 만 건드린다. n_quantiles=1000, output_distribution='normal'
  RTDL_MLPSubSplitInterface
    replace_zero_by_nans   0 -> NaN            (converter 가 미지값을 0 으로 뒀다)
    ord_enc                OrdinalEncoder, 미지/결측 -> -1
  RTDL_MLP.forward
    -1 -> 그 열의 마지막 카테고리 / PLR 임베딩 / 본체 / softmax

수치 전처리(3번 함정)가 이 판본에서 풀렸다. tfms 설정이 ['quantile_tabr'] 이고
s.tfm.tfms[0].tfms.normalizer_ 에 fitted QuantileTransformer 가 들어 있다.
quantiles_ (1000, 44) 와 references_ (1000,) 를 npz 에 담고
sklearn 의 transform 을 그대로 재현한다 (float32 반올림 위치까지 동일).
norm.ppf 는 scipy 없이 Wichura AS241 로 계산한다 (mlp_numpy.ndtri).
"""
import os, pickle, sys, time
import numpy as np
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
from mlp_numpy import mlp_predict, ndtri, quantile_transform     # noqa: E402

# ------------------------------------------------------------------ 원본 열기
m = pickle.load(open(os.path.join(SC, "mlp2024.pkl"), "rb"))
s = m.alg_interface_.sub_split_interfaces[0]
while isinstance(s, (list, tuple)):
    s = s[0]
net = s.model.module_
sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
cat_ind = np.asarray(s.categorical_indicator, dtype=bool)
categories = np.asarray(net.categories, dtype=np.int64)      # 열별 카테고리 수
qt = s.tfm.tfms[0].tfms.normalizer_                          # fitted QuantileTransformer

print(f"범주 열 {int(cat_ind.sum())}개 / 전체 {len(cat_ind)}열 "
      f"(범주 위치 {np.where(cat_ind)[0].min()}~{np.where(cat_ind)[0].max()})")
print(f"열별 카테고리 수 {categories}  합계 {categories.sum()}")
print(f"임베딩 표 {sd['category_embeddings.weight'].shape}")
print(f"분위수 변환 quantiles_ {qt.quantiles_.shape} {qt.quantiles_.dtype}  "
      f"references_ {qt.references_.shape}  출력분포 {qt.output_distribution}")

zc = np.load(os.path.join(SC, "nn_cache.npz"), allow_pickle=False)
Xc_, Xn_, season = zc["Xc"], zc["Xn"], zc["season"]
# 두 경로에 주는 열 순서가 다르다
#   pytabkit  학습 때 준 순서 [범주16 | 수치44]. 내부에서 알아서 재배치한다
#   numpy     이미 재배치된 순서 [수치44 | 범주16] 로 직접 받는다
X_orig = np.concatenate([Xc_.astype(np.float32), Xn_], 1)
X_int = np.concatenate([Xn_, Xc_.astype(np.float32)], 1)
ci = np.where(cat_ind)[0]


# ------------------------------------------------- 범주 매핑을 훅으로 실측한다
# 원값 -> 신경망 인덱스 사이에 OrdinalEncoder 가 두 번 겹쳐 있다.
# 규칙을 코드로 추론하면 틀리기 쉬우므로, 후보값을 실제 파이프라인에 통과시키고
# 신경망 입력 텐서를 훅으로 가로채 매핑표(LUT)를 만든다.
buf = []
h = net.register_forward_pre_hook(lambda mod, inp: buf.append(inp[0].detach().clone().numpy()))


def net_input(X_fit_order):
    """X 를 pytabkit 에 통과시키고 신경망 직전 텐서 (n, 60) 을 돌려준다."""
    buf.clear()
    m.predict_proba(X_fit_order)
    return np.concatenate(buf, 0)


uniq = [np.unique(Xc_[:, j]).astype(np.float64) for j in range(Xc_.shape[1])]
n_probe = max(len(u) for u in uniq) + 1                       # +1 은 미지값 확인용
probe = np.repeat(X_orig[:1], n_probe, axis=0)
for j, u in enumerate(uniq):
    probe[:len(u), j] = u.astype(np.float32)
    probe[len(u):, j] = np.float32(-987654.0)                 # 학습에 없던 값
pin = net_input(probe)

lut_keys, lut_vals, lut_len = [], [], []
for j, u in enumerate(uniq):
    enc = pin[:len(u), ci[j]].astype(np.int64)
    enc = np.where(enc == -1, categories[j] - 1, enc)         # forward 의 -1 치환
    unk = int(pin[len(u):, ci[j]][0])
    assert unk == -1, f"열{j} 미지값이 -1 이 아니다: {unk}"
    lut_keys.append(u); lut_vals.append(enc); lut_len.append(len(u))
    if j < 3:
        print(f"  범주열{j} 고유값 {len(u)}개 -> 인덱스 [{enc.min()}, {enc.max()}] "
              f"(카테고리 수 {categories[j]}, 미지값 -> {categories[j]-1})")
h.remove()

# ------------------------------------------------------------------ 저장
np.savez_compressed(
    os.path.join(SC, "mlp.npz"),
    cat_ind=cat_ind,
    categories=categories,
    cat_offsets=sd["category_offsets"],
    cat_emb=sd["category_embeddings.weight"],
    per_w=sd["num_emb_layer.0.periodic.weight"],
    lin_w=sd["num_emb_layer.0.linear.weight"],
    lin_b=sd["num_emb_layer.0.linear.bias"],
    l0w=sd["layers.0.weight"], l0b=sd["layers.0.bias"],
    l1w=sd["layers.1.weight"], l1b=sd["layers.1.bias"],
    l2w=sd["layers.2.weight"], l2b=sd["layers.2.bias"],
    hw=sd["head.weight"], hb=sd["head.bias"],
    lut_keys=np.concatenate(lut_keys),
    lut_vals=np.concatenate(lut_vals),
    lut_len=np.array(lut_len, dtype=np.int64),
    q_quantiles=qt.quantiles_,
    q_references=qt.references_,
)
P = os.path.join(SC, "mlp.npz")
print(f"\n저장 {P}  ({os.path.getsize(P)/1024/1024:.2f} MB)")
zz = np.load(P, allow_pickle=False)

# ============================================================ 검증 1  ndtri
print("\n" + "=" * 76)
print("검증 1 — ndtri (AS241) vs scipy.stats.norm.ppf")
print("=" * 76)
try:
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    tp = np.concatenate([rng.uniform(0, 1, 2_000_000),
                         rng.uniform(0, 1e-6, 200_000),
                         1 - rng.uniform(0, 1e-6, 200_000),
                         np.array([1e-7, 0.5, 1 - 1e-7])])
    a, b = ndtri(tp), norm.ppf(tp)
    print(f"  n={len(tp):,}  최대절대차 {np.abs(a-b).max():.3e}  "
          f"최대상대차 {np.max(np.abs(a-b)/np.maximum(np.abs(b),1e-300)):.3e}")
except ImportError:
    print("  scipy 없음 — 건너뜀")

# ================================================== 검증 2  분위수 전처리 재현
print("\n" + "=" * 76)
print("검증 2 — 분위수 전처리: numpy 재현 vs sklearn 원본 (float32 비트단위)")
print("=" * 76)
idx2 = np.arange(50_000)
ref_q = qt.transform(Xn_[idx2].astype(np.float32))
got_q = quantile_transform(Xn_[idx2], qt.quantiles_, qt.references_)
same = np.array_equal(ref_q, got_q)
print(f"  n={len(idx2):,}  완전일치={same}  최대차이={np.abs(ref_q.astype(np.float64)-got_q).max():.3e}")

# ================================================ 검증 3  신경망 입력 텐서 전체
print("\n" + "=" * 76)
print("검증 3 — 신경망 입력 텐서 (60열) 재현")
print("=" * 76)
h = net.register_forward_pre_hook(lambda mod, inp: buf.append(inp[0].detach().clone().numpy()))
idx3 = np.arange(20_000)
pin3 = net_input(X_orig[idx3])
h.remove()
mine_num = quantile_transform(X_int[idx3][:, :44], qt.quantiles_, qt.references_)
print(f"  수치 44열 완전일치={np.array_equal(pin3[:, :44], mine_num)}")
mine_cat = np.empty((len(idx3), len(ci)), np.int64)
for j in range(len(ci)):
    v = X_int[idx3][:, 44 + j].astype(np.float64)
    pos = np.clip(np.searchsorted(lut_keys[j], v), 0, len(lut_keys[j]) - 1)
    hit = lut_keys[j][pos] == v
    mine_cat[:, j] = np.where(hit, lut_vals[j][pos], categories[j] - 1)
ref_cat = pin3[:, 44:].astype(np.int64)
ref_cat = np.where(ref_cat == -1, categories[None, :] - 1, ref_cat)
print(f"  범주 16열 완전일치={np.array_equal(ref_cat, mine_cat)}")

# ================================================ 검증 4  최종 예측값
print("\n" + "=" * 76)
print("검증 4 — 최종 예측: numpy 재현 vs pytabkit 원본")
print("=" * 76)
for tag, idx in (("5행", np.arange(5)),
                 ("평가규모", np.where(season == 2024)[0][:245_789])):
    t0 = time.time(); ref = m.predict_proba(X_orig[idx])[:, 1]; t1 = time.time()
    got = mlp_predict(X_int[idx], zz); t2 = time.time()
    d = float(np.max(np.abs(ref - got)))
    # 신경망 본체는 torch 가 float32, 재현본이 float64 라 1e-9 까지는 못 간다.
    # float32 유효자릿수(~1e-7) 안쪽이면 재현 성공으로 본다.
    ok = "PASS" if d < 1e-5 else "FAIL"
    print(f"  {tag:10s} n={len(idx):>7,}  최대차이={d:.3e}  중앙차이="
          f"{np.median(np.abs(ref-got)):.3e}  -> {ok}   "
          f"[pytabkit {t1-t0:.1f}s / numpy {t2-t1:.1f}s]")
    if len(idx) == 5:
        print(f"     pytabkit: {np.round(ref, 8)}")
        print(f"     numpy   : {np.round(got, 8)}")
    else:
        rho = np.corrcoef(ref, got)[0, 1]
        print(f"     상관 {rho:.12f}  평균 {ref.mean():.8f} / {got.mean():.8f}")
