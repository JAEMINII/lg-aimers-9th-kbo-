# -*- coding: utf-8 -*-
"""mlp.npz 를 읽어 순수 numpy 로 MLP-PLR 추론을 한다.

제출 script.py 에 그대로 복사해 넣을 수 있게 이 파일만으로 완결된다.
필요한 것은 numpy 뿐이다 (torch / pytabkit / scipy / sklearn 모두 불필요).

파이프라인 (pytabkit 재현)
  1  수치 44열   QuantileTransformer(n_quantiles=1000, output='normal')
                 보간 -> float32 반올림 -> 경계클립 -> ndtri -> ±5.199 클립 -> float32
  2  범주 16열   원값 -> 정수 인덱스 (LUT). 미지값은 그 열의 마지막 카테고리
  3  PLR 임베딩  2π·w·x -> [cos, sin] -> 피처별 (96->24) 선형 + ReLU
  4  범주 임베딩 category_embeddings[idx + offset]
  5  본체        concat(1056+128=1184) -> 128 -> 256 -> 128 -> 2 -> softmax
"""
import numpy as np

# ---------------------------------------------------------------- ndtri
# 표준정규 분위수함수. scipy 가 없으므로 Wichura AS241 (PPND16) 을 쓴다.
# 파이썬 표준 라이브러리 statistics._normal_dist_inv_cdf 와 같은 알고리즘이라
# scipy.stats.norm.ppf 와 1e-15 수준까지 일치한다 (검증은 export_mlp.py 에서).
_A = (3.38713_28727_96366_6080e+0, 1.33141_66789_17843_7745e+2,
      1.97159_09503_06551_4427e+3, 1.37316_93765_50946_1125e+4,
      4.59219_53931_54987_1457e+4, 6.72657_70927_00870_0853e+4,
      3.34305_75583_58812_8105e+4, 2.50908_09287_30122_6727e+3)
_B = (1.0, 4.23133_30701_60091_1252e+1, 6.87187_00749_20579_0830e+2,
      5.39419_60214_24751_1077e+3, 2.12137_94301_58659_5867e+4,
      3.93078_95800_09271_0610e+4, 2.87290_85735_72194_2674e+4,
      5.22649_52788_52854_5610e+3)
_C = (1.42343_71107_49683_57734e+0, 4.63033_78461_56545_29590e+0,
      5.76949_72214_60691_40550e+0, 3.64784_83247_63204_60504e+0,
      1.27045_82524_52368_38258e+0, 2.41780_72517_74506_11770e-1,
      2.27238_44989_26918_45833e-2, 7.74545_01427_83414_07640e-4)
_D = (1.0, 2.05319_16266_37758_82187e+0, 1.67638_48301_83803_84940e+0,
      6.89767_33498_51000_04550e-1, 1.48103_97642_74800_74590e-1,
      1.51986_66563_61645_71966e-2, 5.47593_80849_95344_94600e-4,
      1.05075_00716_44416_84324e-9)
_E = (6.65790_46435_01103_77720e+0, 5.46378_49111_64114_36990e+0,
      1.78482_65399_17291_33580e+0, 2.96560_57182_85048_91230e-1,
      2.65321_89526_57612_30930e-2, 1.24266_09473_88078_43860e-3,
      2.71155_55687_43487_57815e-5, 2.01033_43992_92288_13265e-7)
_F = (1.0, 5.99832_20655_58879_37690e-1, 1.36929_88092_27358_05310e-1,
      1.48753_61290_85061_48525e-2, 7.86869_13114_56132_59100e-4,
      1.84631_83175_10054_68180e-5, 1.42151_17583_16445_88870e-7,
      2.04426_31033_89939_78564e-15)


def _poly(c, r):
    """호너법. c 는 낮은 차수부터."""
    out = np.full_like(r, c[-1])
    for k in range(len(c) - 2, -1, -1):
        out = out * r + c[k]
    return out


def ndtri(p):
    """표준정규 분위수. p 는 [0, 1]. 0/1 은 ∓inf 를 준다 (뒤에서 클립됨)."""
    p = np.asarray(p, dtype=np.float64)
    q = p - 0.5
    out = np.empty_like(p)

    mid = np.abs(q) <= 0.425
    if mid.any():
        qm = q[mid]
        r = 0.180625 - qm * qm
        out[mid] = qm * _poly(_A, r) / _poly(_B, r)

    tail = ~mid
    if tail.any():
        pt, qt = p[tail], q[tail]
        r = np.where(qt <= 0.0, pt, 1.0 - pt)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.sqrt(-np.log(r))
        near = r <= 5.0
        x = np.empty_like(r)
        with np.errstate(invalid="ignore", over="ignore"):
            # p 가 정확히 0/1 이면 r=inf 라 inf/inf=nan 이 난다. 아래에서 ∓inf 로 덮는다
            if near.any():
                rn = r[near] - 1.6
                x[near] = _poly(_C, rn) / _poly(_D, rn)
            far = ~near
            if far.any():
                rf = r[far] - 5.0
                x[far] = _poly(_E, rf) / _poly(_F, rf)
        out[tail] = np.where(qt < 0.0, -x, x)
    # 경계값. AS241 은 p=0/1 에서 정의되지 않는다. scipy 와 같이 ∓inf 로 두고
    # 호출부의 클립에 맡긴다 (분위수 변환에서 경계는 정확히 0.0/1.0 로 찍힌다)
    out = np.where(p <= 0.0, -np.inf, out)
    out = np.where(p >= 1.0, np.inf, out)
    return out


# ------------------------------------------------- QuantileTransformer 재현
_BOUNDS = 1e-7
_SPACING1 = 2.220446049250313e-16          # np.spacing(1)


def quantile_transform(Xn, q, ref):
    """sklearn QuantileTransformer(output_distribution='normal').transform 재현.

    입력을 float32 로 받아 float32 를 돌려준다. sklearn 이 float32 배열에
    제자리 대입을 하므로 보간 결과가 한 번 float32 로 반올림되는데,
    그 반올림까지 그대로 따라 해야 값이 정확히 일치한다.
    """
    Xn = np.ascontiguousarray(Xn, dtype=np.float32)
    n, d = Xn.shape
    out = np.empty((n, d), dtype=np.float32)
    clip_min = float(ndtri(np.array([_BOUNDS - _SPACING1]))[0])
    clip_max = float(ndtri(np.array([1.0 - (_BOUNDS - _SPACING1)]))[0])
    rq, rr = -q[::-1], -ref[::-1]
    for j in range(d):
        col = Xn[:, j]
        lo_x, hi_x = q[0, j], q[-1, j]
        lo_idx = col - _BOUNDS < lo_x
        hi_idx = col + _BOUNDS > hi_x
        p = 0.5 * (np.interp(col, q[:, j], ref)
                   - np.interp(-col, rq[:, j], rr))
        p32 = p.astype(np.float32)          # sklearn 의 제자리 대입과 같은 반올림
        p32[hi_idx] = np.float32(1.0)
        p32[lo_idx] = np.float32(0.0)
        z = ndtri(p32)
        out[:, j] = np.clip(z, clip_min, clip_max).astype(np.float32)
    return out


# ------------------------------------------------------------------ 추론
def mlp_predict(Xall, z, chunk=8192):
    """Xall: (n, 60) = [수치 44 | 범주 16]  (pytabkit 내부 순서).

    z 는 np.load('mlp.npz') 결과. 반환은 P(성공) 1차원 배열.
    """
    cat_ind = z["cat_ind"]
    ci = np.where(cat_ind)[0]
    ni = np.where(~cat_ind)[0]
    offs, ncat = z["cat_offsets"], z["categories"]
    emb = z["cat_emb"].astype(np.float64)
    per_w = z["per_w"].astype(np.float64)
    lin_w = z["lin_w"].astype(np.float64)
    lin_b = z["lin_b"].astype(np.float64)
    qq, qref = z["q_quantiles"], z["q_references"]
    lk, lv, ll = z["lut_keys"], z["lut_vals"], z["lut_len"]
    keys, vals, k = [], [], 0
    for L in ll:
        keys.append(lk[k:k + L]); vals.append(lv[k:k + L]); k += L
    W = [(z["l0w"].T.astype(np.float64), z["l0b"].astype(np.float64)),
         (z["l1w"].T.astype(np.float64), z["l1b"].astype(np.float64)),
         (z["l2w"].T.astype(np.float64), z["l2b"].astype(np.float64))]
    hw, hb = z["hw"].T.astype(np.float64), z["hb"].astype(np.float64)
    n_num, n_cat, d_emb = len(ni), len(ci), emb.shape[1]
    d_pl = lin_w.shape[2]
    tau = 2.0 * np.pi

    out = np.empty(len(Xall), np.float64)
    for a in range(0, len(Xall), chunk):
        b = min(a + chunk, len(Xall))
        m = b - a
        Xc = Xall[a:b][:, ci]
        Xn = quantile_transform(Xall[a:b][:, ni], qq, qref).astype(np.float64)

        # 범주: 원값 -> 인덱스. LUT 에 없는 값은 마지막 카테고리(=미지)
        idx = np.empty((m, n_cat), np.int64)
        for j in range(n_cat):
            v = Xc[:, j].astype(np.float64)
            pos = np.clip(np.searchsorted(keys[j], v), 0, len(keys[j]) - 1)
            hit = keys[j][pos] == v
            idx[:, j] = np.where(hit, vals[j][pos], ncat[j] - 1)
        e = emb[idx + offs[None, :]].reshape(m, n_cat * d_emb)

        # PLR: 2π·w·x -> [cos, sin] -> 피처별 선형 + ReLU
        h = np.empty((m, n_num, d_pl), np.float64)
        for f in range(n_num):
            zz = (tau * per_w[f])[None, :] * Xn[:, f:f + 1]
            pf = np.concatenate([np.cos(zz), np.sin(zz)], 1)     # (m, 96)
            h[:, f, :] = pf @ lin_w[f] + lin_b[f]
        np.maximum(h, 0, out=h)

        x = np.concatenate([h.reshape(m, n_num * d_pl), e], 1)
        for w, bb in W:
            x = np.maximum(x @ w + bb, 0)
        lg = x @ hw + hb

        # pytabkit 은 softmax -> float32 -> log(+1e-30) -> softmax 를 한 번 더 탄다
        lg = lg - lg.max(1, keepdims=True)
        ex = np.exp(lg)
        p = (ex / ex.sum(1, keepdims=True)).astype(np.float32)
        lp = np.log(p + np.float32(1e-30))
        lp = lp - lp.max(1, keepdims=True)
        ex2 = np.exp(lp.astype(np.float64))
        out[a:b] = ex2[:, 1] / ex2.sum(1)
    return out
