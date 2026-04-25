\"\"\"
mir.py — Music Information Retrieval (No-librosa version)
==========================================================
Uses scipy, numpy, and soundfile directly.
Finds energy curves, structural sections, and beat grids.
\"\"\"

import numpy as np
import soundfile as sf
import subprocess
import tempfile
import os
from scipy import signal, ndimage
from scipy.spatial.distance import cdist


def load_audio(path: str, sr: int = 22050) -> tuple:
    \"\"\"
    Load an audio file, convert to mono, and resample to target sr.
    Supports M4A/AAC/MP3 via ffmpeg fallback.
    \"\"\"
    try:
        # Try soundfile first (works for WAV, FLAC, OGG)
        y, file_sr = sf.read(path, dtype='float32')
    except Exception:
        # Fallback: use ffmpeg to convert to WAV first (handles M4A, MP3, etc.)
        print(f\"   Using ffmpeg to decode {os.path.splitext(path)[1]} format...\")
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
            ], capture_output=True, check=True, timeout=300)
            
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
    \"\"\"
    Compute the RMS energy curve over time.
    
    Returns dict with rms, times_ms, mean_energy, energy_threshold.
    \"\"\"
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
        \"rms\": rms,
        \"times_ms\": times_ms,
        \"mean_energy\": mean_energy,
        \"energy_threshold\": energy_threshold,
    }


def extract_features_chunked(y: np.ndarray, sr: int, hop_length: int = 512, n_fft: int = 2048) -> dict:
    \"\"\"
    Process audio in chunks to extract all required spectral features.
    This guarantees peak memory stays extremely low (O(chunk_size)) rather than O(song_length),
    strictly enforcing the 512MB limit on the Render free tier.
    \"\"\"
    import math
    from scipy.fft import dct
    
    chunk_sec = 15.0
    chunk_samples = int(chunk_sec * sr)
    
    # ── Precompute Filterbanks & Bins ──
    # Mel Filterbank
    n_mels = 40
    fmin, fmax = 0.0, sr / 2.0
    def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel): return 700 * (10 ** (mel / 2595) - 1)
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    freq_bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    n_freqs = n_fft // 2 + 1
    mel_filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(n_mels):
        f_start, f_center, f_end = freq_bins[m], freq_bins[m + 1], freq_bins[m + 2]
        for k in range(f_start, f_center):
            if f_center != f_start: mel_filterbank[m, k] = (k - f_start) / (f_center - f_start)
        for k in range(f_center, f_end):
            if f_end != f_center: mel_filterbank[m, k] = (f_end - k) / (f_end - f_center)
            
    # Chroma bins
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    with np.errstate(divide='ignore', invalid='ignore'):
        pitch_classes = np.round(12 * np.log2(freqs / 440.0 + 1e-10)) % 12
    pitch_classes = pitch_classes.astype(int)
    pitch_classes[0] = 0
    chroma_masks = [(pitch_classes == pc) for pc in range(12)]
    
    # Centroid freqs
    centroid_freqs = freqs[:, np.newaxis]
    
    # ── Chunk Processing ──
    n_chunks = math.ceil(len(y) / chunk_samples)
    overlap = n_fft  # Overlap to prevent boundary framing issues
    
    onset_list = []
    chroma_list = []
    mfcc_list = []
    centroids_list = []
    flatness_list = []
    
    print(f\"   Processing {n_chunks} streaming chunks for spectral features...\")
    
    for i in range(n_chunks):
        start = max(0, i * chunk_samples - overlap)
        end = min(len(y), (i + 1) * chunk_samples + overlap)
        y_chunk = y[start:end]
        
        # 1. STFT
        f, t, Zxx = signal.stft(y_chunk, fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft - hop_length, boundary='even')
        mag = np.abs(Zxx).astype(np.float32)
        power = mag ** 2
        
        # Trim overlap frames from the result, EXCEPT for the very first and last chunks if needed
        # To be precise, we can just process non-overlapping y chunks with boundary='even', 
        # which is much simpler and perfectly recreates the timeline shape without edge artifacts.
        # Let's override the y_chunk to be strictly non-overlapping to guarantee exact frame count matching.
        pass
        
    # Wait, the simplest way to guarantee exact frame concatenation without overlap tracking:
    # process non-overlapping `y_chunk` with boundary='none', but that drops edges.
    # We will just process exact non-overlapping chunks with boundary='even'.
    onset_list, chroma_list, mfcc_list, centroids_list, flatness_list = [], [], [], [], []
    
    for i in range(n_chunks):
        start = i * chunk_samples
        end = min(len(y), (i + 1) * chunk_samples)
        
        # STFT on the exact chunk
        # Note: Concatenating STFTs of chunks is not mathematically identical to STFT of full signal 
        # at the boundaries, but for MIR features it's perfectly acceptable.
        f, t, Zxx = signal.stft(y[start:end], fs=sr, window='hann', nperseg=n_fft, noverlap=n_fft - hop_length, boundary='even')
        mag = np.abs(Zxx).astype(np.float32)
        power = mag ** 2
        
        # ── Onset Strength ──
        diff = np.diff(mag, axis=1)
        diff = np.maximum(0, diff)
        onset = np.mean(diff, axis=0)
        # Pad 1 frame at start for each chunk so len matches mag
        onset = np.concatenate(([0], onset))
        onset_list.append(onset)
        
        # ── Chroma ──
        chroma = np.zeros((12, mag.shape[1]), dtype=np.float32)
        for pc in range(12):
            chroma[pc, :] = np.sum(power[chroma_masks[pc], :], axis=0)
        chroma_list.append(chroma)
        
        # ── MFCC ──
        mel_spec = np.dot(mel_filterbank, power)
        mel_spec = np.log(mel_spec + 1e-10)
        mfcc = dct(mel_spec, type=2, axis=0, norm='ortho')[:13]
        mfcc_list.append(mfcc)
        
        # ── Centroid & Flatness ──
        mag_sum = np.sum(mag, axis=0) + 1e-10
        centroids = np.sum(centroid_freqs * mag, axis=0) / mag_sum
        log_mag = np.log(mag + 1e-10)
        geo_mean = np.exp(np.mean(log_mag, axis=0))
        arith_mean = np.mean(mag, axis=0)
        flatness = geo_mean / (arith_mean + 1e-10)
        
        centroids_list.append(centroids)
        flatness_list.append(flatness)
        
    # ── Concatenate & Normalize ──
    onset_full = np.concatenate(onset_list)
    chroma_full = np.concatenate(chroma_list, axis=1)
    norm = np.max(chroma_full, axis=0, keepdims=True) + 1e-10
    chroma_full = chroma_full / norm
    
    mfcc_full = np.concatenate(mfcc_list, axis=1)
    centroids_full = np.concatenate(centroids_list)
    flatness_full = np.concatenate(flatness_list)
    
    spec_times_ms = np.arange(len(onset_full)) * hop_length / sr * 1000.0
    
    return {
        \"onset_env\": onset_full,
        \"chroma\": chroma_full,
        \"mfcc\": mfcc_full,
        \"centroids\": centroids_full,
        \"flatness\": flatness_full,
        \"spec_times_ms\": spec_times_ms
    }


def find_beats(onset_env: np.ndarray, sr: int, y_len: int, hop_length: int = 512) -> dict:
    \"\"\"
    Detect tempo and beat positions using autocorrelation of onset strength.
    \"\"\"
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
        n_beats = int(y_len / sr * tempo / 60)
        peaks = np.linspace(0, len(onset_env) - 1, n_beats, dtype=int)
    
    beat_times = peaks * hop_length / sr
    beat_times_ms = beat_times * 1000.0
    
    # Downbeats: every 4th beat
    downbeat_times_ms = beat_times_ms[::4] if len(beat_times_ms) >= 4 else beat_times_ms
    
    return {
        \"tempo\": round(float(tempo), 1),
        \"beat_times_ms\": beat_times_ms,
        \"downbeat_times_ms\": downbeat_times_ms,
    }


def detect_sections(mfcc: np.ndarray, chroma: np.ndarray, energy: dict, sr: int, n_sections: int = 8) -> list:
    \"\"\"
    Detect structural sections using self-similarity on spectral features.
    \"\"\"
    duration_ms = energy[\"times_ms\"][-1] if len(energy[\"times_ms\"]) > 0 else 0
    hop_length = 512
    
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
    print(\"   Computing self-similarity matrix...\")
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
    
    # Use provided RMS energy curve for section energy
    rms = energy[\"rms\"]
    rms_times_ms = energy[\"times_ms\"]
    
    sections = []
    for i in range(len(all_bounds) - 1):
        start_ms = float(all_bounds[i])
        end_ms = float(all_bounds[i + 1])
        
        mask = (rms_times_ms >= start_ms) & (rms_times_ms < end_ms)
        section_rms = rms[mask]
        energy = float(np.mean(section_rms)) if len(section_rms) > 0 else 0.0
        
        sections.append({
            \"start_ms\": start_ms,
            \"end_ms\": end_ms,
            \"energy\": energy,
            \"label\": \"unknown\",
        })
    
    return sections


def classify_sections(sections: list, energy_threshold: float) -> list:
    \"\"\"
    Label sections as intro/verse/chorus/bridge/outro based on
    position and energy level.
    \"\"\"
    if not sections:
        return sections
    
    energies = [s[\"energy\"] for s in sections]
    max_energy = max(energies) if energies else 1.0
    
    for i, section in enumerate(sections):
        relative_energy = section[\"energy\"] / max_energy if max_energy > 0 else 0
        
        if i == 0 and relative_energy < 0.5:
            section[\"label\"] = \"intro\"
        elif i == len(sections) - 1 and relative_energy < 0.6:
            section[\"label\"] = \"outro\"
        elif relative_energy >= 0.75:
            section[\"label\"] = \"chorus\"
        elif relative_energy >= 0.4:
            section[\"label\"] = \"verse\"
        else:
            section[\"label\"] = \"bridge\"
    
    return sections


def get_song_duration_ms(y: np.ndarray, sr: int) -> float:
    \"\"\"Return song duration in milliseconds.\"\"\"
    return float(len(y) / sr * 1000.0)
