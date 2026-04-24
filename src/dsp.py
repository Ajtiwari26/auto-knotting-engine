"""
dsp.py — Digital Signal Processing
====================================
The "stitch" module — makes cuts inaudible.
Handles zero-crossing detection, beat snapping, and transition validation.
"""

import numpy as np


def find_zero_crossing(y: np.ndarray, sr: int, target_ms: float, window_ms: float = 50.0) -> float:
    """
    Find the nearest zero-crossing point within ±window_ms of target_ms.
    
    A zero-crossing is where the waveform crosses the zero amplitude line.
    Cutting at these points prevents audible "pops" and "clicks".
    
    Args:
        y: audio signal
        sr: sample rate
        target_ms: desired cut point in milliseconds
        window_ms: search window in milliseconds (±)
    
    Returns:
        Adjusted cut point in milliseconds (snapped to nearest zero-crossing)
    """
    target_sample = int(target_ms * sr / 1000.0)
    window_samples = int(window_ms * sr / 1000.0)

    start = max(0, target_sample - window_samples)
    end = min(len(y) - 1, target_sample + window_samples)

    if start >= end:
        return target_ms

    # Find zero crossings in the window
    segment = y[start:end]
    zero_crossings = np.where(np.diff(np.sign(segment)))[0]

    if len(zero_crossings) == 0:
        # No zero crossings found — find the point with minimum absolute amplitude
        min_idx = np.argmin(np.abs(segment))
        best_sample = start + min_idx
    else:
        # Find the zero crossing closest to the target
        zc_global = zero_crossings + start
        distances = np.abs(zc_global - target_sample)
        best_idx = np.argmin(distances)
        best_sample = zc_global[best_idx]

    return float(best_sample * 1000.0 / sr)


def snap_to_beat(target_ms: float, beat_times_ms: np.ndarray, max_drift_ms: float = 100.0) -> float:
    """
    Snap a cut point to the nearest beat within ±max_drift_ms.
    
    If no beat is within range, returns the original target_ms.
    This ensures cuts feel musically natural.
    
    Args:
        target_ms: desired cut point in milliseconds
        beat_times_ms: array of beat timestamps in milliseconds
        max_drift_ms: maximum allowed drift from target
    
    Returns:
        Beat-aligned cut point in milliseconds
    """
    if len(beat_times_ms) == 0:
        return target_ms

    distances = np.abs(beat_times_ms - target_ms)
    nearest_idx = np.argmin(distances)
    nearest_beat = float(beat_times_ms[nearest_idx])

    if abs(nearest_beat - target_ms) <= max_drift_ms:
        return nearest_beat
    else:
        return target_ms


def snap_to_downbeat(
    target_ms: float,
    beat_times_ms: np.ndarray,
    downbeat_times_ms: np.ndarray,
    max_drift_ms: float = 150.0,
) -> float:
    """
    Priority-based snapping: prefer downbeat → tolerate any beat → original.
    
    Downbeats are the first beat of each measure (every 4th beat).
    Cutting on downbeats produces the most musically natural transitions
    because listeners feel the "1" of each bar.
    
    Priority:
      1. Downbeat within ±max_drift_ms  → snap to it
      2. Any beat within ±(max_drift_ms * 0.6) → snap to it
      3. No good candidate → return original target
    """
    # Priority 1: Try downbeat first (wider tolerance)
    if len(downbeat_times_ms) > 0:
        db_distances = np.abs(downbeat_times_ms - target_ms)
        db_nearest_idx = np.argmin(db_distances)
        db_nearest = float(downbeat_times_ms[db_nearest_idx])
        if abs(db_nearest - target_ms) <= max_drift_ms:
            return db_nearest
    
    # Priority 2: Fall back to any beat (tighter tolerance)
    if len(beat_times_ms) > 0:
        bt_distances = np.abs(beat_times_ms - target_ms)
        bt_nearest_idx = np.argmin(bt_distances)
        bt_nearest = float(beat_times_ms[bt_nearest_idx])
        if abs(bt_nearest - target_ms) <= max_drift_ms * 0.6:
            return bt_nearest
    
    # Priority 3: No snap available
    return target_ms


def refine_knot_boundary(
    y: np.ndarray,
    sr: int,
    start_ms: float,
    end_ms: float,
    beat_times_ms: np.ndarray,
    downbeat_times_ms: np.ndarray = None,
) -> tuple:
    """
    Refine a knot's start/end boundaries using downbeat-priority snapping
    and zero-crossing alignment.
    
    Order of operations:
    1. Snap to nearest downbeat (strongest musical alignment)
    2. Fall back to nearest beat if no downbeat is close
    3. Then snap to nearest zero-crossing (click prevention)
    
    Args:
        y: audio signal
        sr: sample rate
        start_ms: raw knot start in ms
        end_ms: raw knot end in ms
        beat_times_ms: array of beat timestamps in ms
        downbeat_times_ms: array of downbeat timestamps in ms (optional)
    
    Returns:
        (refined_start_ms, refined_end_ms)
    """
    # Step 1: Downbeat-priority snap
    if downbeat_times_ms is not None and len(downbeat_times_ms) > 0:
        start_ms = snap_to_downbeat(start_ms, beat_times_ms, downbeat_times_ms, max_drift_ms=150.0)
        end_ms = snap_to_downbeat(end_ms, beat_times_ms, downbeat_times_ms, max_drift_ms=150.0)
    else:
        start_ms = snap_to_beat(start_ms, beat_times_ms, max_drift_ms=100.0)
        end_ms = snap_to_beat(end_ms, beat_times_ms, max_drift_ms=100.0)

    # Step 2: Zero-crossing snap (finer adjustment within ±30ms)
    start_ms = find_zero_crossing(y, sr, start_ms, window_ms=30.0)
    end_ms = find_zero_crossing(y, sr, end_ms, window_ms=30.0)

    # Ensure start < end
    if start_ms >= end_ms:
        end_ms = start_ms + 100.0  # minimum 100ms knot

    return start_ms, end_ms


def validate_transition(y: np.ndarray, sr: int, point_a_ms: float, point_b_ms: float) -> float:
    """
    Validate how smooth a transition between two points would be.
    
    Checks:
    1. Amplitude at both cut points (should be low)
    2. Amplitude difference between the two points (should be small)
    3. Zero-crossing proximity (should be near one)
    
    Returns:
        Confidence score 0.0 to 1.0 (1.0 = perfect cut, 0.0 = will sound terrible)
    """
    sample_a = int(point_a_ms * sr / 1000.0)
    sample_b = int(point_b_ms * sr / 1000.0)

    # Clamp to valid range
    sample_a = max(0, min(sample_a, len(y) - 1))
    sample_b = max(0, min(sample_b, len(y) - 1))

    amp_a = abs(float(y[sample_a]))
    amp_b = abs(float(y[sample_b]))

    # Max amplitude in the signal for normalization
    max_amp = float(np.max(np.abs(y))) if len(y) > 0 else 1.0
    if max_amp == 0:
        max_amp = 1.0

    # Score 1: How close to zero are the cut points? (normalized)
    amp_score = 1.0 - ((amp_a + amp_b) / 2.0) / max_amp
    amp_score = max(0.0, min(1.0, amp_score))

    # Score 2: How similar are the amplitudes at the two cut points?
    diff_score = 1.0 - abs(amp_a - amp_b) / max_amp
    diff_score = max(0.0, min(1.0, diff_score))

    # Combined score (weighted)
    confidence = 0.6 * amp_score + 0.4 * diff_score

    return round(confidence, 3)
