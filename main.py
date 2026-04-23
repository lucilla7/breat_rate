import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, detrend

# ==========================================
# CONSTANTS
# ==========================================
FS = 16  # Sampling frequency (Hz)
WINDOW_SEC = 60  # Analysis window length (s) - 60s for better resolution
STEP_SEC = 30  # Output interval (s) - Results reported every 30s
LOWCUT = 0.13  # ~8 bpm (Lower respiration limit)
HIGHCUT = 0.45  # ~27 bpm (Upper respiration limit)


# ==========================================
# SIGNAL PROCESSING
# ==========================================
def bandpass_filter(data: np.ndarray):
    """Applies a 4th order Butterworth bandpass filter."""
    nyq = 0.5 * FS
    low = LOWCUT / nyq
    high = HIGHCUT / nyq
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data)


def preprocess_signal(data: np.ndarray):
    """Removes linear trend and applies bandpass filtering."""
    data = detrend(data, type='linear')
    return bandpass_filter(data)


# ==========================================
# RESPIRATION RATE ESTIMATION
# ==========================================
def estimate_rr_from_segment(seg: np.ndarray):
    """Estimates Breathing Rate using both Frequency (Welch) and Time (ACF) domains."""
    # 1. Welch Method (Frequency Domain)
    f, pxx = welch(seg, fs=FS, nperseg=len(seg))
    mask = (f >= LOWCUT) & (f <= HIGHCUT)
    if not np.any(mask):
        return np.nan
    rr_welch = f[mask][np.argmax(pxx[mask])] * 60

    # 2. Autocorrelation Method (Time Domain)
    # Highlight periodicities by calculating ACF
    acf = np.correlate(seg - np.mean(seg), seg - np.mean(seg), mode='full')
    acf = acf[len(acf) // 2:]
    lag_min, lag_max = int(FS / HIGHCUT), int(FS / LOWCUT)

    # Identify the highest peak within the physiological respiratory range
    lag = np.argmax(acf[lag_min:lag_max]) + lag_min
    rr_acf = 60 * FS / lag

    # Logic Fusion: If both methods agree (within 3 bpm), average them;
    # otherwise, default to ACF which is typically more robust to noise.
    if abs(rr_welch - rr_acf) < 3:
        return (rr_welch + rr_acf) / 2
    return rr_acf


def calculate_confidence(seg, seg_raw, corr):
    """Calculates a quality score based on spectral purity and motion detection."""
    f, pxx = welch(seg, fs=FS, nperseg=len(seg))
    band_power = np.sum(pxx[(f >= LOWCUT) & (f <= HIGHCUT)])
    total_power = np.sum(pxx)

    # Spectral Purity: ratio of power in the respiratory band vs total power
    spectral_purity = band_power / total_power if total_power > 0 else 0

    # Motion Penalty: reduce confidence if raw signal variance is too high (artifact detection)
    motion_penalty = 1.0 if np.std(seg_raw) < 1.0 else 0.4

    # Combine metrics including inter-channel correlation
    return np.clip(spectral_purity * motion_penalty * ((corr + 1) / 2), 0, 1)


# ==========================================
# MAIN PIPELINE
# ==========================================
def main_pipeline(file_path):
    # Load data
    df = pd.read_csv(file_path, header=None)
    ch1, ch2 = df.iloc[:, 0].values, df.iloc[:, 1].values

    # --- Visualization of Raw vs Filtered Signals ---
    def plot_signal_comparison(ch1, ch2, duration_sec=120):
        n_samples = duration_sec * FS
        time = np.arange(n_samples) / FS

        # Take the first segment
        s1_raw = ch1[:n_samples]
        s2_raw = ch2[:n_samples]

        # Apply the same preprocessing used in the pipeline
        s1_filt = preprocess_signal(s1_raw)
        s2_filt = preprocess_signal(s2_raw)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot Raw
        axes[0].plot(time, s1_raw, label='Raw Ch 1', alpha=0.7)
        axes[0].plot(time, s2_raw, label='Raw Ch 2', alpha=0.7)
        axes[0].set_ylabel('Amplitude (Raw)')
        axes[0].set_title('Raw Bio-Motion Signals')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot Filtered
        axes[1].plot(time, s1_filt, label='Filtered Ch 1', color='tab:blue')
        axes[1].plot(time, s2_filt, label='Filtered Ch 2', color='tab:orange')
        axes[1].set_ylabel('Amplitude (Filtered)')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_title(f'Filtered Respiration Signals ({LOWCUT}-{HIGHCUT} Hz)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Plot signals:
    plot_signal_comparison(ch1, ch2)

    results = []
    win_samples = WINDOW_SEC * FS
    step_samples = STEP_SEC * FS

    # Sliding window with overlap
    for start in range(0, len(ch1) - win_samples + 1, step_samples):
        s1_raw = ch1[start: start + win_samples]
        s2_raw = ch2[start: start + win_samples]

        # Preprocessing
        s1 = preprocess_signal(s1_raw)
        s2 = preprocess_signal(s2_raw)

        # Channel Fusion: Average channels if highly correlated,
        # otherwise pick the one with higher energy (signal strength).
        corr = np.corrcoef(s1, s2)[0, 1]
        if corr > 0.7:
            best_seg = (s1 + s2) / 2
        else:
            best_seg = s1 if np.std(s1) > np.std(s2) else s2

        # Estimation and Confidence
        rr = estimate_rr_from_segment(best_seg)
        conf = calculate_confidence(best_seg, (s1_raw + s2_raw) / 2, corr)

        # Quality Filter: Reject outliers or low confidence segments
        if rr < 8 or rr > 25 or conf < 0.3:
            rr = np.nan

        results.append({
            'time_s': start / FS,
            'rr_bpm': rr,
            'confidence': conf
        })

    res_df = pd.DataFrame(results)

    # Post-processing: Interpolate gaps (max 2 consecutive) and smooth the curve
    res_df['rr_bpm_final'] = res_df['rr_bpm'].interpolate(limit=2).rolling(window=3, center=True).mean()

    return res_df


# ==========================================
# PIPELINE AND PLOTTING
# ==========================================
if __name__ == "__main__":
    # Run the processing pipeline
    results_df = main_pipeline("signals.csv")

    # Save to CSV
    results_df.to_csv("respiration_results.csv", index=False)
    print("Processing complete. Results saved in 'respiration_results.csv'.")

    # Plotting
    plt.figure(figsize=(12, 8))

    # Top Plot: Respiration Rate
    plt.subplot(2, 1, 1)
    plt.plot(results_df['time_s'] / 60, results_df['rr_bpm'], 'ko', alpha=0.2, label='Raw Estimates')
    plt.plot(results_df['time_s'] / 60, results_df['rr_bpm_final'], 'r-', linewidth=2, label='Smoothed RR')
    plt.ylabel("Breathing Rate (bpm)")
    plt.title("Estimated Respiration Rate Over Time")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Bottom Plot: Confidence Score
    plt.subplot(2, 1, 2)
    plt.fill_between(results_df['time_s'] / 60, results_df['confidence'], color='green', alpha=0.3)
    plt.plot(results_df['time_s'] / 60, results_df['confidence'], color='green', label='Confidence')
    plt.axhline(0.3, color='black', linestyle='--', label='Rejection Threshold')
    plt.ylabel("Confidence Score (0-1)")
    plt.xlabel("Time (minutes)")
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.show()



