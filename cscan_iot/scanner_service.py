import os, time, math, threading
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from printer_control_v2 import setup_precision_printer
from hs5_control import HS5StreamPeaks
from cloud_manager import CloudManager  # <--- NEW IMPORT

# --- Helper Functions (No changes here) ---
def row_from_pulses_nosmooth(aa, ncols, expected_cycles, center_window=True):
    out = np.full(ncols, np.nan, dtype=np.float32)
    m = int(aa.size)
    if m == 0 or ncols <= 0: return out
    W = int(max(8, min(expected_cycles, m)))
    start = max(0, (m - W) // 2) if center_window else 0
    core = aa[start:start + W].astype(np.float32)
    valid = np.isfinite(core)
    if not np.any(valid): return out
    core = core[valid]; W = core.size
    if W == 0: return out
    x = (np.arange(W, dtype=np.float64) + 0.5) * (ncols / W)
    i0 = np.floor(x).astype(np.int64)
    w1 = x - i0; w0 = 1.0 - w1
    i1 = i0 + 1
    sumw = np.zeros(ncols, dtype=np.float64)
    acc  = np.zeros(ncols, dtype=np.float64)
    m0 = (i0 >= 0) & (i0 < ncols)
    if np.any(m0):
        np.add.at(sumw, i0[m0], w0[m0]); np.add.at(acc, i0[m0], w0[m0]*core[m0])
    m1 = (i1 >= 0) & (i1 < ncols)
    if np.any(m1):
        np.add.at(sumw, i1[m1], w1[m1]); np.add.at(acc, i1[m1], w1[m1]*core[m1])
    nz = sumw > 1e-12
    out[nz] = (acc[nz] / sumw[nz]).astype(np.float32)
    for k in range(1, ncols):
        if np.isnan(out[k]) and not np.isnan(out[k-1]): out[k] = out[k-1]
    return out

def align_row_minblur(row, ref, max_shift=8.0, prev_shift=0.0):
    if ref is None or not np.any(np.isfinite(ref)): return row, 0.0
    a = np.nan_to_num(row, nan=0.0); a -= np.mean(a)
    b = np.nan_to_num(ref,  nan=0.0); b -= np.mean(b)
    n = len(a)
    fa = np.fft.rfft(a); fb = np.fft.rfft(b)
    R = fa * np.conj(fb); R /= np.maximum(np.abs(R), 1e-12)
    c = np.fft.irfft(R, n=n)
    k0 = int(np.argmax(c))
    denom = (c[(k0-1)%n] - 2*c[k0] + c[(k0+1)%n])
    delta = 0.5*(c[(k0-1)%n] - c[(k0+1)%n])/denom if abs(denom)>1e-12 else 0.0
    shift_est = float(k0 + delta)
    if shift_est > n/2: shift_est -= n
    shift_est = float(np.clip(shift_est, -max_shift, max_shift))
    mask = np.ones_like(c, dtype=bool)
    lo, hi = (k0-5)%n, (k0+5)%n
    if lo<=hi: mask[lo:hi+1]=False
    else: mask[:hi+1]=False; mask[lo:]=False
    if np.std(c[mask]) == 0: psr = 0
    else: psr = (c[k0] - np.mean(c[mask])) / (np.std(c[mask]) + 1e-12)
    if psr < 6.0 or abs(shift_est - prev_shift) > 1.8:
        shift_use = prev_shift
    else:
        shift_use = shift_est
    k = np.fft.rfftfreq(n)
    row_out = np.fft.irfft(np.fft.rfft(np.nan_to_num(row, nan=0.0)) * np.exp(-2j*np.pi*k*shift_use), n=n)
    return row_out.astype(np.float32), shift_use

# --- The Service Class ---
class CScanService:
    def __init__(self):
        self.running = False
        self.stop_signal = False
        self.thread = None
        self.status = "IDLE"
        self.progress = {"line": 0, "total": 0, "msg": "Ready"}
        self.config = {
            "roi_w": 50.0, "roi_h": 50.0, "pitch": 0.1, "speed": 10.0,
            "cols": 500, "out_dir": "cscan_out", "cmap": "turbo"
        }
        self.images = {"Amplitude": None, "ToF": None, "Energy": None}
        
        # Initialize Cloud
        self.cloud = CloudManager()

    def start_scan(self, new_config=None):
        if self.running: return False, "Already Running"
        if new_config: self.config.update(new_config)
        self.stop_signal = False
        self.running = True
        self.status = "RUNNING"
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()
        return True, "Started"

    def stop_scan(self):
        if self.running:
            self.stop_signal = True
            self.status = "STOPPING"
            return True
        return False

    def _save_plot(self, img, name, label):
        """Saves locally AND uploads to cloud"""
        if not np.any(np.isfinite(img)): return
        
        # 1. Save Locally
        local_path = os.path.join(self.config["out_dir"], name)
        cm = matplotlib.colormaps.get_cmap(self.config["cmap"]).copy()
        cm.set_bad('white')
        
        finite = img[np.isfinite(img)]
        vmin, vmax = np.percentile(finite, [5, 95]) if finite.size >= 16 else (0, 1)
        
        plt.figure(figsize=(8, 6), dpi=100)
        plt.imshow(img, cmap=cm, origin="upper", aspect="equal", 
                   extent=[0, self.config["roi_w"], self.config["roi_h"], 0],
                   vmin=vmin, vmax=vmax, interpolation="nearest")
        plt.title(f"{label} (Line {self.progress['line']}/{self.progress['total']})")
        plt.colorbar(); plt.tight_layout()
        plt.savefig(local_path, bbox_inches='tight')
        plt.close()
        
        # 2. Upload to Cloud
        cloud_url = self.cloud.upload_image_async(local_path, label)
        
        if cloud_url:
            self.images[label] = cloud_url  # Use AWS URL
        else:
            self.images[label] = f"/local/{name}" # Fallback to local

    def _worker(self):
        cfg = self.config
        os.makedirs(cfg["out_dir"], exist_ok=True)
        nlines = max(2, int(math.ceil(cfg["roi_h"] / cfg["pitch"])))
        self.progress["total"] = nlines
        
        img_amp = np.full((nlines, cfg["cols"]), np.nan, dtype=np.float32)
        img_tof = np.full((nlines, cfg["cols"]), np.nan, dtype=np.float32)
        img_eng = np.full((nlines, cfg["cols"]), np.nan, dtype=np.float32)
        
        pr, hs = None, None
        try:
            self.progress["msg"] = "Initializing Hardware..."
            pr, xl, xr, ys, ye = setup_precision_printer("COM6", 115200, cfg["roi_w"], cfg["roi_h"])
            hs = HS5StreamPeaks(fs_hz=20_000_000, feature_mode="envelope").open()
            hs.calibrate_sync(seconds=1.0, verbose=False)
            
            theo_time = abs(xr - xl) / cfg["speed"]
            detected_prf = getattr(hs, "detected_prf", 5000.0)
            if detected_prf <= 0: detected_prf = 5000.0
            expected_cycles = int(detected_prf * theo_time)
            
            ref_e, ref_o = None, None
            sh_e, sh_o = 0.0, 0.0

            for i in range(nlines):
                if self.stop_signal: break
                self.progress["line"] = i + 1
                self.progress["msg"] = f"Scanning Line {i+1}..."
                
                y = ys + i * cfg["pitch"]
                ltr = (i % 2 == 0)
                x0, x1 = (xl, xr) if ltr else (xr, xl)
                
                pr.move_to_position(x0, y, fast=True)
                pr.wait_for_completion(); time.sleep(0.05)
                pr.send_command(f"G1 X{x1:.3f} F{int(cfg['speed']*60)}")
                
                t0 = time.perf_counter()
                tt, aa, tf, ee = hs.acquire_peaks(duration_s=theo_time + 0.3)
                pr.wait_for_completion()
                t1 = time.perf_counter()
                
                sel = (tt >= t0) & (tt <= t1)
                aa, tf, ee = aa[sel], tf[sel], ee[sel]
                if not ltr: aa, tf, ee = aa[::-1], tf[::-1], ee[::-1]
                
                ra = row_from_pulses_nosmooth(aa, cfg["cols"], expected_cycles)
                rf = row_from_pulses_nosmooth(tf, cfg["cols"], expected_cycles)
                re = row_from_pulses_nosmooth(ee, cfg["cols"], expected_cycles)
                
                if ltr:
                    ra, sh_e = align_row_minblur(ra, ref_e, prev_shift=sh_e)
                    rf, _ = align_row_minblur(rf, ref_e, prev_shift=sh_e)
                    re, _ = align_row_minblur(re, ref_e, prev_shift=sh_e)
                    ref_e = ra if ref_e is None else ref_e
                else:
                    ra, sh_o = align_row_minblur(ra, ref_o, prev_shift=sh_o)
                    rf, _ = align_row_minblur(rf, ref_o, prev_shift=sh_o)
                    re, _ = align_row_minblur(re, ref_o, prev_shift=sh_o)
                    ref_o = ra if ref_o is None else ref_o

                img_amp[i,:] = ra
                img_tof[i,:] = rf
                img_eng[i,:] = re
                
                if i % 5 == 0 or i == nlines - 1:
                    self._save_plot(img_amp, "scan_amp.png", "Amplitude")
                    self._save_plot(img_tof, "scan_tof.png", "ToF")
                    self._save_plot(img_eng, "scan_eng.png", "Energy")
            
            if not self.stop_signal:
                self.progress["msg"] = "Returning to Start..."
                pr.move_to_position(xl, ys, fast=True)
                pr.wait_for_completion()

            self.status = "COMPLETED"
            self.progress["msg"] = "Scan Finished."
            
        except Exception as e:
            self.status = "ERROR"
            self.progress["msg"] = f"Error: {str(e)}"
            print(f"Error: {e}")
            import traceback; traceback.print_exc()
        finally:
            if hs: hs.close()
            if pr: pr.close()
            self.running = False