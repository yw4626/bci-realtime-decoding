from __future__ import annotations

from typing import List, Literal, Sequence, Tuple

import numpy as np

WindowSampling = Literal["sequential", "uniform", "stratified"]

# ECoG 常用划分（Hz），在 rFFT 上按 [f_lo, f_hi) 对每通道取平均功率；可按课题在 extract_window_features 中扩展。
DEFAULT_PHYSIOLOGICAL_BANDS_HZ: Tuple[Tuple[str, float, float], ...] = (
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low_gamma", 30.0, 70.0),
    ("high_gamma", 70.0, 150.0),
)


def _band_power_like(window: np.ndarray) -> np.ndarray:
    spec = np.fft.rfft(window, axis=0)
    power = np.abs(spec) ** 2
    return _legacy_band_power_from_power(power)


def _legacy_band_power_from_power(power: np.ndarray) -> np.ndarray:
    low = power[1:8].mean(axis=0)
    mid = power[8:25].mean(axis=0)
    high = power[25:80].mean(axis=0)
    return np.concatenate([low, mid, high], axis=0)


def _band_power_hz(
    window: np.ndarray,
    sample_rate_hz: float,
    f_lo: float,
    f_hi: float,
) -> np.ndarray:
    """单窗内各通道在 [f_lo, f_hi) Hz 上的平均 rFFT 功率（不含 DC）。"""
    window = np.asarray(window, dtype=np.float64)
    if window.ndim == 1:
        window = window.reshape(-1, 1)
    n_t = int(window.shape[0])
    n_ch = int(window.shape[1])
    if n_t < 2:
        return np.zeros(n_ch, dtype=np.float64)
    spec = np.fft.rfft(window, axis=0)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(n_t, d=1.0 / float(sample_rate_hz))
    mask = (freqs >= f_lo) & (freqs < f_hi)
    if not np.any(mask):
        return np.zeros(n_ch, dtype=np.float64)
    return power[mask].mean(axis=0).astype(np.float64, copy=False)


def _physiological_band_powers_from_power(
    power: np.ndarray,
    freqs: np.ndarray,
    bands_hz: Sequence[Tuple[str, float, float]],
) -> np.ndarray:
    """由单次 rFFT 的功率谱与频率轴计算各生理频带平均功率（与逐带调用 _band_power_hz 数值一致）。"""
    n_ch = int(power.shape[1])
    parts: List[np.ndarray] = []
    for _, lo, hi in bands_hz:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            parts.append(np.zeros(n_ch, dtype=np.float64))
        else:
            parts.append(power[mask].mean(axis=0).astype(np.float64, copy=False))
    return np.concatenate(parts, axis=0)


def _physiological_band_powers(
    window: np.ndarray,
    sample_rate_hz: float,
    bands_hz: Sequence[Tuple[str, float, float]] = DEFAULT_PHYSIOLOGICAL_BANDS_HZ,
) -> np.ndarray:
    """与 _band_power_like 相同排布：每个频带一段 shape (n_ch,)，再沿轴 0 拼接。"""
    window = np.asarray(window, dtype=np.float64)
    if window.ndim == 1:
        window = window.reshape(-1, 1)
    n_t = int(window.shape[0])
    n_ch = int(window.shape[1])
    if n_t < 2:
        return np.zeros(n_ch * len(bands_hz), dtype=np.float64)
    spec = np.fft.rfft(window, axis=0)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(n_t, d=1.0 / float(sample_rate_hz))
    return _physiological_band_powers_from_power(power, freqs, bands_hz)


def extract_window_features(
    window: np.ndarray,
    sample_rate_hz: int = 1000,
    *,
    include_legacy_fft_bands: bool = False,
    include_physiological_bands: bool = True,
    physiological_bands_hz: Sequence[Tuple[str, float, float]] = DEFAULT_PHYSIOLOGICAL_BANDS_HZ,
) -> np.ndarray:
    """
    时域：每通道 mean / std / RMS。
    频域：可选旧版三段 rFFT 索引分带；可选按 Hz 的 alpha / beta / low-gamma / high-gamma 等平均功率。
    关闭所有频域项时仅保留时域三段（与早期极简设定接近）。
    """
    mean_f = window.mean(axis=0)
    std_f = window.std(axis=0)
    rms_f = np.sqrt(np.mean(window**2, axis=0))
    chunks: List[np.ndarray] = [mean_f, std_f, rms_f]

    need_fft = include_legacy_fft_bands or include_physiological_bands
    w64 = np.asarray(window, dtype=np.float64)
    if w64.ndim == 1:
        w64 = w64.reshape(-1, 1)
    n_t = int(w64.shape[0])
    n_ch = int(w64.shape[1])

    spec_power: np.ndarray | None = None
    freqs: np.ndarray | None = None
    if need_fft and n_t >= 2:
        spec = np.fft.rfft(w64, axis=0)
        spec_power = np.abs(spec) ** 2
        freqs = np.fft.rfftfreq(n_t, d=1.0 / float(sample_rate_hz))

    if include_legacy_fft_bands:
        if spec_power is None:
            chunks.append(_band_power_like(w64))
        else:
            chunks.append(_legacy_band_power_from_power(spec_power))

    if include_physiological_bands:
        if spec_power is None or freqs is None:
            chunks.append(np.zeros(n_ch * len(physiological_bands_hz), dtype=np.float64))
        else:
            chunks.append(
                _physiological_band_powers_from_power(spec_power, freqs, physiological_bands_hz)
            )

    return np.concatenate(chunks, axis=0)


def _pick_window_positions(
    n_avail: int,
    max_windows: int,
    sampling: WindowSampling,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    在 ``n_avail`` 个候选滑窗位置中选出 ``max_windows`` 个下标（0..n_avail-1）。
    ``sequential`` 与旧行为一致：只取时间轴最前面的一段（长数据+限额时后半段永不进训练）。
    """
    if max_windows <= 0 or n_avail <= max_windows:
        return np.arange(n_avail, dtype=np.int64)
    if sampling == "sequential":
        return np.arange(max_windows, dtype=np.int64)
    if sampling == "uniform":
        return np.sort(rng.choice(n_avail, size=max_windows, replace=False))

    # stratified：沿时间轴分桶，每桶随机抽若干，避免样本全堆在录音开头
    n_bins = min(32, max(4, max_windows // 8))
    per = max_windows // n_bins
    picks: List[np.ndarray] = []
    edges = np.linspace(0, n_avail, n_bins + 1).astype(np.int64)
    for b in range(n_bins):
        lo, hi = int(edges[b]), int(edges[b + 1])
        chunk = np.arange(lo, hi, dtype=np.int64)
        if chunk.size == 0:
            continue
        take = min(per, int(chunk.size))
        picks.append(rng.choice(chunk, size=take, replace=False))
    if not picks:
        return np.sort(rng.choice(n_avail, size=max_windows, replace=False))
    idx = np.concatenate(picks)
    if idx.size < max_windows:
        remaining = np.setdiff1d(np.arange(n_avail, dtype=np.int64), idx, assume_unique=False)
        if remaining.size > 0:
            need = int(max_windows - idx.size)
            extra = rng.choice(remaining, size=min(need, int(remaining.size)), replace=False)
            idx = np.concatenate([idx, extra])
    if idx.size > max_windows:
        idx = rng.choice(idx, size=max_windows, replace=False)
    return np.sort(idx)


def build_supervised_windows(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
    step_size: int,
    max_windows: int = 0,
    sampling: WindowSampling = "stratified",
    random_state: int = 42,
    sample_rate_hz: int = 1000,
    include_legacy_fft_bands: bool = False,
    include_physiological_bands: bool = True,
    physiological_bands_hz: Sequence[Tuple[str, float, float]] = DEFAULT_PHYSIOLOGICAL_BANDS_HZ,
) -> Tuple[np.ndarray, np.ndarray]:
    ends = np.arange(window_size, len(x), step_size, dtype=np.int64)
    n_avail = int(ends.size)
    if n_avail == 0:
        return np.zeros((0, 0)), np.zeros((0, y.shape[1]))

    rng = np.random.default_rng(random_state)
    pos_idx = _pick_window_positions(
        n_avail,
        max_windows if max_windows > 0 else n_avail,
        sampling if max_windows > 0 and n_avail > max_windows else "sequential",
        rng,
    )
    chosen_ends = ends[pos_idx]

    feats = []
    targets = []
    for end in chosen_ends:
        e = int(end)
        start = e - window_size
        window = x[start:e]
        feats.append(
            extract_window_features(
                window,
                sample_rate_hz=sample_rate_hz,
                include_legacy_fft_bands=include_legacy_fft_bands,
                include_physiological_bands=include_physiological_bands,
                physiological_bands_hz=physiological_bands_hz,
            )
        )
        targets.append(y[e - 1])
    return np.asarray(feats), np.asarray(targets)

