"""
mir.py — Music Information Retrieval (No-librosa version)
==========================================================
Uses scipy, numpy, and soundfile directly.
Finds energy curves, structural sections, and beat grids.
"""

import numpy as np
import soundfile as sf
import subprocess
import tempfile
import os
from scipy import signal, ndimage
from scipy.spatial.distance import cdist


def load_audio(path: str, sr: int = 22050) -> tuple:
    """
    Load an audio file, convert to mono, and resample to target sr.
    Supports M4A/AAC/MP3 via ffmpeg fallback.
    """
    try:
        # Try soundfile first (works for WAV, FLAC, OGG)
        y, file_sr = sf.read(path, dtype='float32')
    except Exception:
        # Fallback: use ffmpeg to convert to WAV first (handles M4A, MP3, etc.)
        print(f"   Using ffmpeg to decode {os.path.splitext(path)[1]} format...")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            subprocess.run([
                'ffmpeg', '-i', path,
                '-ar', str(sr),
                '-ac', '1',  # mono
                '-f', 'wav',
                '-y',  # overwrite
                tmp_path
            ], capture_output=True, check=True, timeout=60)
            
            y, file_sr = sf.read(tmp_path, dtype='float32')
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    
    # Resample if needed
    if file_sr != sr:
        num_samples = int(len(y) * sr / file_sr)
        y = signal.resample(y, num_samples)
    
    return y, sr


def compute_energy_curve(y: np.ndarray, sr: int, hop_length: int = 512, frame_length: int = 2048) -> dict:
    """
    Compute the RMS energy curve over time.
    
    Returns dict with rms, times_ms, mean_energy, energy_threshold.
    """
    # Compute RMS in frames
    n_frames = 1 + (len(y) - frame_length) // hop_length
    if n_frames <= 0:
        n_frames = 1
    
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        end = min(start + frame_length, len(y))
        frame = y[start:end]
        rms[i] = np.sqrt(np.mean(frame ** 2))
    
    times = np.arange(n_frames) * hop_length / sr
    times_ms = times * 1000.0
    
    mean_energy = float(np.mean(rms))
    energy_threshold = mean_energy * 0.6
    
    return {
        "rms": rms,
        "times_ms": times_ms,
        "mean_energy": mean_energy,
        "energy_threshold": energy_threshold,
    }


def _onset_strength(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """Compute onset strength envelope using spectral flux with O(1) vectorized STFT."""
    n_fft = 2048
    
    # Vectorized STFT (blazing fast, ~100x faster than pure python loops)
    # boundary='even' is equivalent to mode='reflect' in np.pad
    f, t, Zxx = signal.stft(y, fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft - hop_length, boundary='even')
    mag = np.abs(Zxx).astype(np.float32)
    
    # Spectral flux (difference between consecutive frames)
    diff = np.diff(mag, axis=1)
    diff = np.maximum(0, diff)
    
    onset = np.mean(diff, axis=0)
    
    # np.diff reduces length by 1, pad at the beginning to maintain frame count
    onset = np.concatenate(([0], onset))
    
    return onset


def find_beats(y: np.ndarray, sr: int, hop_length: int = 512) -> dict:
    """
    Detect tempo and beat positions using autocorrelation of onset strength.
    
    Returns dict with tempo, beat_times_ms, downbeat_times_ms.
    """
    onset_env = _onset_strength(y, sr, hop_length)
    
    # Tempo estimation via autocorrelation
    # Search range: 60-200 BPM
    min_bpm, max_bpm = 60, 200
    min_lag = int(60.0 / max_bpm * sr / hop_length)
    max_lag = int(60.0 / min_bpm * sr / hop_length)
    max_lag = min(max_lag, len(onset_env) - 1)
    
    if max_lag <= min_lag:
        # Song is very short, just estimate
        tempo = 120.0
    else:
        # Autocorrelation
        acf = np.correlate(onset_env, onset_env, mode='full')
        acf = acf[len(onset_env) - 1:]  # positive lags only
        
        # Find peak in tempo range
        search_region = acf[min_lag:max_lag + 1]
        if len(search_region) > 0:
            best_lag = min_lag + np.argmax(search_region)
            tempo = 60.0 * sr / (best_lag * hop_length)
        else:
            tempo = 120.0
    
    # Beat tracking via peak picking on onset envelope
    # Expected beat period in frames
    beat_period = int(60.0 / tempo * sr / hop_length)
    
    if beat_period < 1:
        beat_period = 1
    
    # Smooth the onset envelope
    if len(onset_env) > beat_period:
        kernel_size = max(3, beat_period // 2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        onset_smooth = ndimage.median_filter(onset_env, size=kernel_size)
    else:
        onset_smooth = onset_env
    
    # Peak picking with minimum distance = beat_period * 0.5
    min_distance = max(1, int(beat_period * 0.5))
    peaks, _ = signal.find_peaks(onset_smooth, distance=min_distance, height=np.mean(onset_smooth) * 0.3)
    
    if len(peaks) == 0:
        # Fallback: generate evenly-spaced beats
        n_beats = int(len(y) / sr * tempo / 60)
        peaks = np.linspace(0, len(onset_env) - 1, n_beats, dtype=int)
    
    beat_times = peaks * hop_length / sr
    beat_times_ms = beat_times * 1000.0
    
    # Downbeats: every 4th beat
    downbeat_times_ms = beat_times_ms[::4] if len(beat_times_ms) >= 4 else beat_times_ms
    
    return {
        "tempo": round(float(tempo), 1),
        "beat_times_ms": beat_times_ms,
        "downbeat_times_ms": downbeat_times_ms,
    }


def _compute_chroma(y: np.ndarray, sr: int, hop_length: int = 512, n_fft: int = 2048) -> np.ndarray:
    """Compute chroma features from audio using vectorized STFT."""
    f, t, Zxx = signal.stft(y, fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft - hop_length, boundary='even')
    power_spectrum = np.abs(Zxx) ** 2
    
    n_frames = power_spectrum.shape[1]
    chroma = np.zeros((12, n_frames))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    
    # Map frequencies to chroma bins
    # A4 = 440Hz reference
    with np.errstate(divide='ignore', invalid='ignore'):
        pitch_classes = np.round(12 * np.log2(freqs / 440.0 + 1e-10)) % 12
    pitch_classes = pitch_classes.astype(int)
    pitch_classes[0] = 0  # DC component
    
    # Vectorized binning: for each pitch class, sum the corresponding frequency bins across all frames
    for pc in range(12):
        mask = pitch_classes == pc
        chroma[pc, :] = np.sum(power_spectrum[mask, :], axis=0)
    
    # Normalize
    norm = np.max(chroma, axis=0, keepdims=True) + 1e-10
    chroma = chroma / norm
    
    return chroma


def _compute_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13, hop_length: int = 512, n_fft: int = 2048) -> np.ndarray:
    """Compute MFCCs from audio using mel filterbank with vectorized STFT."""
    f, t, Zxx = signal.stft(y, fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft - hop_length, boundary='even')
    power_spectrum = np.abs(Zxx) ** 2
    
    n_frames = power_spectrum.shape[1]
    
    # Mel filterbank
    n_mels = 40
    fmin, fmax = 0.0, sr / 2.0
    
    # Mel scale conversion
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    freq_bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    # Create filterbank
    n_freqs = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freqs))
    for m in range(n_mels):
        f_start = freq_bins[m]
        f_center = freq_bins[m + 1]
        f_end = freq_bins[m + 2]
        
        for k in range(f_start, f_center):
            if f_center != f_start:
                filterbank[m, k] = (k - f_start) / (f_center - f_start)
        for k in range(f_center, f_end):
            if f_end != f_center:
                filterbank[m, k] = (f_end - k) / (f_end - f_center)
    
    # Apply mel filterbank using fast matrix multiplication
    mel_spec = np.dot(filterbank, power_spectrum)
    
    # Log and DCT to get MFCCs
    mel_spec = np.log(mel_spec + 1e-10)
    
    # DCT-II (simplified)
    from scipy.fft import dct
    mfcc = dct(mel_spec, type=2, axis=0, norm='ortho')[:n_mfcc]
    
    return mfcc


def detect_sections(y: np.ndarray, sr: int, n_sections: int = 8) -> list:
    """
    Detect structural sections using self-similarity on spectral features.
    
    Uses MFCC + chroma features, builds a self-similarity matrix,
    and applies agglomerative-style segmentation.
    """
    duration_ms = len(y) / sr * 1000.0
    hop_length = 512
    
    # Compute features
    print("   Computing MFCC features...")
    mfcc = _compute_mfcc(y, sr, n_mfcc=13, hop_length=hop_length)
    
    print("   Computing chroma features...")
    chroma = _compute_chroma(y, sr, hop_length=hop_length)
    
    # Ensure same number of frames
    min_frames = min(mfcc.shape[1], chroma.shape[1])
    mfcc = mfcc[:, :min_frames]
    chroma = chroma[:, :min_frames]
    
    # Stack features
    features = np.vstack([mfcc, chroma])  # (25, n_frames)
    
    # Reduce resolution by averaging over blocks for faster processing
    block_size = max(1, min_frames // 200)  # ~200 blocks
    n_blocks = min_frames // block_size
    
    if n_blocks < n_sections:
        n_blocks = min_frames
        block_size = 1
    
    blocked_features = np.zeros((features.shape[0], n_blocks))
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, min_frames)
        blocked_features[:, i] = np.mean(features[:, start:end], axis=1)
    
    # Normalize features
    feat_norm = np.linalg.norm(blocked_features, axis=0, keepdims=True) + 1e-10
    blocked_features = blocked_features / feat_norm
    
    # Self-similarity matrix (cosine similarity)
    print("   Computing self-similarity matrix...")
    S = 1 - cdist(blocked_features.T, blocked_features.T, metric='cosine')
    
    # Novelty curve: detect section boundaries using a checkerboard kernel
    kernel_size = max(4, n_blocks // 20)
    if kernel_size % 2 != 0:
        kernel_size += 1
    
    novelty = np.zeros(n_blocks)
    half_k = kernel_size // 2
    
    for i in range(half_k, n_blocks - half_k):
        # Checkerboard kernel on the self-similarity matrix
        tl = S[i - half_k:i, i - half_k:i]  # top-left (same segment)
        br = S[i:i + half_k, i:i + half_k]    # bottom-right (same segment)
        tr = S[i - half_k:i, i:i + half_k]    # top-right (cross segment)
        bl = S[i:i + half_k, i - half_k:i]    # bottom-left (cross segment)
        
        # High novelty when cross-segment similarity is low
        within = (np.mean(tl) + np.mean(br)) / 2
        across = (np.mean(tr) + np.mean(bl)) / 2
        novelty[i] = max(0, within - across)
    
    # Find peaks in novelty curve = section boundaries
    min_section_blocks = max(1, n_blocks // (n_sections * 2))
    peaks, _ = signal.find_peaks(novelty, distance=min_section_blocks,
                                  height=np.mean(novelty) * 0.5)
    
    # Limit to n_sections - 1 boundaries (most prominent)
    if len(peaks) > n_sections - 1:
        # Keep the strongest peaks
        peak_heights = novelty[peaks]
        sorted_idx = np.argsort(peak_heights)[::-1][:n_sections - 1]
        peaks = np.sort(peaks[sorted_idx])
    
    # Convert block indices to time in ms
    bound_times_ms = peaks * block_size * hop_length / sr * 1000.0
    
    # Build section list
    all_bounds = np.concatenate([[0], bound_times_ms, [duration_ms]])
    all_bounds = np.sort(np.unique(all_bounds))
    
    # Compute RMS energy curve for section energy
    rms_data = compute_energy_curve(y, sr, hop_length=hop_length)
    rms = rms_data["rms"]
    rms_times_ms = rms_data["times_ms"]
    
    sections = []
    for i in range(len(all_bounds) - 1):
        start_ms = float(all_bounds[i])
        end_ms = float(all_bounds[i + 1])
        
        mask = (rms_times_ms >= start_ms) & (rms_times_ms < end_ms)
        section_rms = rms[mask]
        energy = float(np.mean(section_rms)) if len(section_rms) > 0 else 0.0
        
        sections.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "energy": energy,
            "label": "unknown",
        })
    
    return sections


def classify_sections(sections: list, energy_threshold: float) -> list:
    """
    Label sections as intro/verse/chorus/bridge/outro based on
    position and energy level.
    """
    if not sections:
        return sections
    
    energies = [s["energy"] for s in sections]
    max_energy = max(energies) if energies else 1.0
    
    for i, section in enumerate(sections):
        relative_energy = section["energy"] / max_energy if max_energy > 0 else 0
        
        if i == 0 and relative_energy < 0.5:
            section["label"] = "intro"
        elif i == len(sections) - 1 and relative_energy < 0.6:
            section["label"] = "outro"
        elif relative_energy >= 0.75:
            section["label"] = "chorus"
        elif relative_energy >= 0.4:
            section["label"] = "verse"
        else:
            section["label"] = "bridge"
    
    return sections


def get_song_duration_ms(y: np.ndarray, sr: int) -> float:
    """Return song duration in milliseconds."""
    return float(len(y) / sr * 1000.0)
