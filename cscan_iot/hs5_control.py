# -*- coding: utf-8 -*-
"""
TiePie HS5 control (STREAM + SYNC + Feature Extraction)
Extracts: Amplitude, ToF, and Pulse Energy
"""
import time
import numpy as np

try:
    import libtiepie
except ImportError as e:
    raise SystemExit("Missing dependency: pip install python-libtiepie") from e


def _envelope_hilbert_numpy(x: np.ndarray) -> np.ndarray:
    """Hilbert envelope without SciPy dependency."""
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    if n == 0: return x
    X = np.fft.rfft(x)
    H = np.zeros_like(X)
    if n % 2 == 0:
        H[0] = 1.0; H[1:-1] = 2.0; H[-1] = 1.0
    else:
        H[0] = 1.0; H[1:] = 2.0
    xa = np.fft.irfft(X * H, n=n)
    return np.abs(xa).astype(np.float32, copy=False)


class HS5StreamPeaks:
    def __init__(self, fs_hz=20_000_000, win_us=60.0, gate_us=(30.0, 40.0),
                 ch1_range=1.0, ch1_coupling="AC",
                 sync_thresh_guess=0.0, min_hyst_v=0.02,
                 prf_hint_hz=5000.0,
                 feature_mode: str = "envelope",
                 p2p_percentiles: tuple[int, int] = (5, 95)):
        
        self.fs = float(fs_hz)
        self.dt = 1.0 / self.fs
        self.win_us = float(win_us)
        self.g0 = max(0, int(round(gate_us[0] * 1e-6 * self.fs)))
        self.g1 = max(self.g0 + 1, int(round(gate_us[1] * 1e-6 * self.fs)))
        self.rec_len = max(10_000, int(0.0005 * self.fs))

        self.range = float(ch1_range)
        self.coupling = ch1_coupling.upper()

        self.sync_thresh_guess = float(sync_thresh_guess)
        self.min_hyst_v = float(min_hyst_v)
        self.prf_hz = float(prf_hint_hz)

        # Refractory period (0.5 * Period)
        T = 1.0 / self.prf_hz if self.prf_hz > 0 else 2e-4
        self.refractory_samp = max(1, int(round(0.5 * T * self.fs)))

        self.ch2_lo = self.sync_thresh_guess - self.min_hyst_v
        self.ch2_hi = self.sync_thresh_guess + self.min_hyst_v
        self.detected_prf = self.prf_hz

        self.feature_mode = (feature_mode or "envelope")
        self.p2p_percentiles = p2p_percentiles
        self.scp = None

    def _feature_value(self, x: np.ndarray) -> float:
        """Calculate scalar amplitude based on mode."""
        if x is None or x.size == 0: return np.nan
        v = x.astype(np.float32, copy=False)
        v = v - np.mean(v, dtype=np.float64) # DC removal

        mode = (self.feature_mode or "envelope").lower()
        if mode == "p2p":
            return float(np.max(v) - np.min(v))
        if mode == "p2p_robust":
            lo, hi = self.p2p_percentiles if self.p2p_percentiles else (5, 95)
            qlo = np.percentile(v, lo)
            qhi = np.percentile(v, hi)
            return float(qhi - qlo)

        # Default: Envelope Peak
        env = _envelope_hilbert_numpy(v)
        return float(np.max(env))

    def _compute_features(self, x: np.ndarray) -> tuple[float, float, float]:
        """
        Calculates:
        1. Amplitude (based on feature_mode)
        2. ToF (Time within gate to peak envelope)
        3. Energy (Sum of squared samples)
        """
        if x is None or x.size == 0:
            return (np.nan, np.nan, np.nan)
        
        v = x.astype(np.float32, copy=False)
        v_centered = v - np.mean(v, dtype=np.float64)
        
        # 1. Amplitude
        amp = self._feature_value(v)
        
        # 2. ToF (using Envelope peak)
        env = _envelope_hilbert_numpy(v_centered)
        pk_idx = int(np.argmax(env)) if env.size else 0
        tof_us = (pk_idx / self.fs) * 1e6
        
        # 3. Energy (Sum of Squares)
        # This is proportional to the total energy of the pulse in the gate
        energy = float(np.sum(v_centered * v_centered))
        
        return (float(amp), float(tof_us), float(energy))

    def open(self):
        libtiepie.network.auto_detect_enabled = True
        libtiepie.device_list.update()
        self.scp = None
        for it in libtiepie.device_list:
            if it.can_open(libtiepie.DEVICETYPE_OSCILLOSCOPE):
                s = it.open_oscilloscope()
                if s.measure_modes & libtiepie.MM_STREAM:
                    self.scp = s; break
        if self.scp is None:
            raise RuntimeError("HS5 not found")

        s = self.scp
        s.measure_mode = libtiepie.MM_STREAM
        s.sample_rate = min(self.fs, s.sample_rate_max)
        self.fs = s.sample_rate; self.dt = 1.0 / self.fs

        target = max(self.rec_len, 5 * self.g1, 50_000)
        s.record_length = min(int(target), s.record_length_max)
        self.rec_len = s.record_length

        for ch in s.channels: ch.enabled = False
        ch1 = s.channels[0]; ch1.enabled = True
        ch1.range = float(self.range)
        ch1.coupling = libtiepie.CK_ACV if (self.coupling == "AC") else libtiepie.CK_DCV
        ch2 = s.channels[1]; ch2.enabled = True
        ch2.range = 5.0; ch2.coupling = libtiepie.CK_DCV

        print(f"[HS5] Opened. fs={self.fs/1e6:.1f}MHz. Gate=[{self.g0}:{self.g1}]")
        return self

    def close(self):
        if self.scp:
            if self.scp.is_running: self.scp.stop()
            self.scp = None

    def calibrate_sync(self, seconds=1.0, verbose=True):
        s = self.scp
        if s.is_running: s.stop()
        s.start()
        
        t0 = time.perf_counter()
        ch2_all = []
        while (time.perf_counter() - t0) < float(seconds):
            if not s.is_data_ready: time.sleep(0.0005); continue
            data = s.get_data()
            if data and len(data) >= 2:
                ch2_all.append(np.asarray(data[1], dtype=np.float32))
        s.stop()
        
        if not ch2_all: raise RuntimeError("Calibration failed: No data")
        ch2 = np.concatenate(ch2_all)
        
        vmin, vmax = np.percentile(ch2, 5), np.percentile(ch2, 95)
        span = max(vmax - vmin, self.min_hyst_v * 2)
        vmid = 0.5 * (vmin + vmax)
        self.ch2_lo = vmid - 0.2 * span
        self.ch2_hi = vmid + 0.2 * span
        
        # Estimate PRF
        above = ch2 > self.ch2_hi
        edges = np.flatnonzero((above[1:] & ~above[:-1]))
        if edges.size >= 2:
             self.detected_prf = self.fs / np.median(np.diff(edges))
        
        # Recalculate Refractory
        T = 1.0 / max(self.detected_prf, 1.0)
        self.refractory_samp = max(1, int(round(0.5 * T * self.fs)))

        if verbose:
            print(f"[CAL] Thresholds: {self.ch2_lo:.2f}/{self.ch2_hi:.2f}V. Detected PRF: {self.detected_prf:.1f}Hz")

    def acquire_peaks(self, duration_s=1.0):
        s = self.scp
        if s.is_running: s.stop()
        s.start()

        thr = float(self.ch2_hi)
        refr = int(self.refractory_samp)
        g0, g1 = self.g0, self.g1
        t0 = time.perf_counter()
        cum_abs = 0
        last_edge_abs = -1e12

        carry_ch1 = np.empty(0, dtype=np.float32)
        carry_ch2 = np.empty(0, dtype=np.float32)

        p_amp, p_tof, p_eng, p_tt = [], [], [], []

        try:
            while (time.perf_counter() - t0) < duration_s:
                if not s.is_data_ready: time.sleep(0.0005); continue
                data = s.get_data()
                if not data: continue
                
                c1 = np.asarray(data[0], dtype=np.float32)
                c2 = np.asarray(data[1], dtype=np.float32)
                n_new = min(c1.size, c2.size)
                if n_new == 0: continue
                
                c1 = c1[:n_new]; c2 = c2[:n_new]
                blk_start = cum_abs
                cum_abs += n_new

                if carry_ch1.size:
                    c1 = np.concatenate((carry_ch1, c1))
                    c2 = np.concatenate((carry_ch2, c2))
                    combo_start = blk_start - carry_ch1.size
                else:
                    combo_start = blk_start

                # Find edges
                mask = (c2[:-1] < thr) & (c2[1:] >= thr)
                edges = np.flatnonzero(mask)

                for e in edges:
                    e_abs = combo_start + e
                    if (e_abs - last_edge_abs) < refr: continue
                    
                    s0, s1 = e + g0, e + g1
                    if s1 > c1.size: continue # Wait for next block
                    
                    # Extract Features
                    amp, tof, eng = self._compute_features(c1[s0:s1])
                    
                    p_amp.append(amp)
                    p_tof.append(tof)
                    p_eng.append(eng)
                    p_tt.append(t0 + e_abs/self.fs)
                    last_edge_abs = e_abs

                # Carry over tail
                rem = min(g1, c1.size)
                carry_ch1 = c1[-rem:]; carry_ch2 = c2[-rem:]

        finally:
            if s.is_running: s.stop()

        if not p_tt:
            return np.array([]), np.array([]), np.array([]), np.array([])
            
        tt = np.array(p_tt, dtype=np.float64)
        aa = np.array(p_amp, dtype=np.float32)
        tf = np.array(p_tof, dtype=np.float32)
        ee = np.array(p_eng, dtype=np.float32)
        
        idx = np.argsort(tt)
        return tt[idx], aa[idx], tf[idx], ee[idx]