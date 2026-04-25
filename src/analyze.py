#!/usr/bin/env python3
"""
analyze.py — Auto-Knotting Engine CLI (v2 — Deep Analysis)
============================================================
Multi-layer knotting pipeline:

  Layer 1: STRUCTURAL — Intro/Outro trimming (section labels)
  Layer 2: ENERGY VALLEYS — Detect sustained low-energy dips inside sections
  Layer 3: VOCAL GAPS — Detect long instrumental-only passages via spectral centroid
  Layer 4: REPETITION — Detect repeated chorus/verse and skip redundant repeats

Each layer independently proposes knot candidates.
Final stage merges, deduplicates, and applies minimum-duration guardrails.

Usage:
  python3 src/analyze.py \\
    --file "test-samples/Song.m4a" \\
    --sensitivity balanced \\
    --output output/result.json
"""

import argparse
import json
import os
import sys
import time
import subprocess
import shutil

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mir import (
    load_audio,
    compute_energy_curve,
    detect_sections,
    classify_sections,
    find_beats,
    get_song_duration_ms,
    extract_features_chunked,
)
from dsp import refine_knot_boundary, validate_transition


# ══════════════════════════════════════════════════════════════
# LAYER 1: STRUCTURAL (intro/outro/bridge)
# ══════════════════════════════════════════════════════════════
def _layer_structural(sections: list, sensitivity: str) -> list:
    """Skip intros, outros, and long low-energy bridges."""
    knots = []
    for section in sections:
        label = section["label"]
        dur_ms = section["end_ms"] - section["start_ms"]

        if sensitivity == "aggressive":
            if label == "intro":
                knots.append({"start_ms": section["start_ms"], "end_ms": section["end_ms"],
                              "reason": f"intro ({dur_ms/1000:.1f}s)", "layer": "structural"})
            elif label == "outro":
                knots.append({"start_ms": section["start_ms"], "end_ms": section["end_ms"],
                              "reason": f"outro ({dur_ms/1000:.1f}s)", "layer": "structural"})
            elif label == "bridge" and dur_ms > 5000:
                knots.append({"start_ms": section["start_ms"], "end_ms": section["end_ms"],
                              "reason": f"bridge ({dur_ms/1000:.1f}s)", "layer": "structural"})
        elif sensitivity == "balanced":
            if label == "intro" and dur_ms > 8000:
                knots.append({"start_ms": section["start_ms"], "end_ms": section["end_ms"],
                              "reason": f"long intro ({dur_ms/1000:.1f}s)", "layer": "structural"})
            elif label == "outro":
                knots.append({"start_ms": section["start_ms"], "end_ms": section["end_ms"],
                              "reason": f"outro ({dur_ms/1000:.1f}s)", "layer": "structural"})
            elif label == "bridge" and dur_ms > 8000:
                knots.append({"start_ms": section["start_ms"], "end_ms": section["end_ms"],
                              "reason": f"low-energy bridge ({dur_ms/1000:.1f}s)", "layer": "structural"})
        else:  # light
            if label == "intro" and dur_ms > 15000:
                knots.append({"start_ms": section["start_ms"], "end_ms": section["end_ms"],
                              "reason": f"long intro ({dur_ms/1000:.1f}s)", "layer": "structural"})

    return knots


# ══════════════════════════════════════════════════════════════
# LAYER 2: ENERGY VALLEYS — find sustained quiet dips mid-song
# ══════════════════════════════════════════════════════════════
def _layer_energy_valleys(energy: dict,
                          duration_ms: float, sensitivity: str) -> list:
    """
    Scan the RMS energy curve for sustained low-energy valleys.
    
    A "valley" is a contiguous region where energy drops below a 
    dynamic threshold for longer than min_duration. These are typically:
    - Musical pauses between verses
    - Instrumental lulls / breathing room
    - Quiet pre-chorus build-ups that add nothing
    """
    rms = energy["rms"]
    times_ms = energy["times_ms"]
    
    if len(rms) < 10:
        return []
    
    # Smooth the energy curve to avoid micro-fluctuation false positives
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(rms, size=25)
    
    # Dynamic threshold: valleys are below a % of the song's peak energy
    peak_energy = np.percentile(smoothed, 90)  # use 90th percentile, not max (avoids outliers)
    mean_energy = np.mean(smoothed)
    
    if sensitivity == "aggressive":
        valley_thresh = mean_energy * 0.65   # anything below 65% of mean
        min_valley_ms = 3000                  # 3s minimum
    elif sensitivity == "balanced":
        valley_thresh = mean_energy * 0.45    # below 45% of mean
        min_valley_ms = 4000                  # 4s minimum
    else:  # light
        valley_thresh = mean_energy * 0.30    # below 30% of mean
        min_valley_ms = 6000                  # 6s minimum
    
    # Also ensure valley threshold is meaningfully below the peak
    valley_thresh = min(valley_thresh, peak_energy * 0.35)
    
    # ── Build-up protection: compute energy slope ──
    # If a valley has a steadily increasing slope leading into a
    # high-energy section (chorus/drop), it's a tension build-up.
    # Cutting it would ruin the musical anticipation.
    energy_gradient = np.gradient(smoothed)
    # Smooth the gradient to avoid micro-fluctuations
    smooth_gradient = uniform_filter1d(energy_gradient, size=30)
    
    # Scan for contiguous regions below threshold
    below = smoothed < valley_thresh
    knots = []
    
    i = 0
    while i < len(below):
        if below[i]:
            start_i = i
            while i < len(below) and below[i]:
                i += 1
            end_i = i - 1
            
            start_ms = float(times_ms[start_i])
            end_ms = float(times_ms[min(end_i, len(times_ms) - 1)])
            valley_dur = end_ms - start_ms
            
            # Check it's long enough and not at the very start/end
            # (structural layer handles those)
            is_interior = start_ms > duration_ms * 0.08 and end_ms < duration_ms * 0.92
            
            if valley_dur >= min_valley_ms and is_interior:
                # ── BUILD-UP PROTECTION ──
                # Check if the second half of this valley has rising energy
                # (i.e., it's a build-up, not a dead zone)
                mid_i = (start_i + end_i) // 2
                second_half_grad = smooth_gradient[mid_i:end_i + 1]
                
                # If >60% of the second half has positive slope, it's a build-up
                if len(second_half_grad) > 0:
                    rising_fraction = np.mean(second_half_grad > 0)
                    avg_slope = float(np.mean(second_half_grad))
                    
                    # Also check if what follows the valley is high-energy
                    lookahead = min(end_i + 50, len(smoothed) - 1)
                    post_valley_energy = float(np.mean(smoothed[end_i:lookahead]))
                    is_buildup = (rising_fraction > 0.60 and 
                                  avg_slope > 0 and 
                                  post_valley_energy > peak_energy * 0.6)
                    
                    if is_buildup:
                        # Skip this valley — it's a crescendo build-up
                        continue
                
                avg_valley_energy = float(np.mean(smoothed[start_i:end_i + 1]))
                knots.append({
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "reason": f"energy valley ({valley_dur/1000:.1f}s, energy={avg_valley_energy:.4f})",
                    "layer": "energy_valley",
                })
        i += 1
    
    return knots


# ══════════════════════════════════════════════════════════════
# LAYER 3: VOCAL GAP DETECTION — find long instrumental passages
# ══════════════════════════════════════════════════════════════
def _layer_vocal_gaps(centroids: np.ndarray, flatness: np.ndarray, spec_times_ms: np.ndarray,
                      energy: dict, duration_ms: float, sensitivity: str) -> list:
    """
    Detect long instrumental-only passages using spectral features.
    """
    print("   Computing spectral features for vocal gap detection...")
    rms = energy["rms"]
    
    if len(centroids) < 20:
        return []
    
    # Smooth the centroid and flatness
    from scipy.ndimage import uniform_filter1d
    smooth_centroid = uniform_filter1d(centroids, size=40)
    smooth_flatness = uniform_filter1d(flatness, size=40)
    
    # Build a "vocal likelihood" score per frame
    # Vocals typically have centroid > 1000 Hz and moderate flatness
    # Normalize centroid to 0-1 range
    c_min, c_max = np.percentile(smooth_centroid, 5), np.percentile(smooth_centroid, 95)
    c_range = c_max - c_min if c_max > c_min else 1.0
    c_norm = (smooth_centroid - c_min) / c_range
    c_norm = np.clip(c_norm, 0, 1)
    
    # Also use RMS energy — low energy + low centroid = definitely instrumental gap
    rms_interp = np.interp(spec_times_ms, energy["times_ms"], rms)
    rms_norm = rms_interp / (np.percentile(rms_interp, 90) + 1e-10)
    rms_norm = np.clip(rms_norm, 0, 1)
    
    # Combined "vocal activity" score
    vocal_score = 0.5 * c_norm + 0.3 * rms_norm + 0.2 * smooth_flatness
    
    # Smooth it further to get a clean signal
    vocal_smooth = uniform_filter1d(vocal_score, size=50)
    
    # Threshold for "instrumental" (low vocal activity)
    if sensitivity == "aggressive":
        vocal_thresh = np.percentile(vocal_smooth, 45)  # bottom 45%
        min_gap_ms = 4000
    elif sensitivity == "balanced":
        vocal_thresh = np.percentile(vocal_smooth, 35)  # bottom 35%
        min_gap_ms = 5000
    else:  # light
        vocal_thresh = np.percentile(vocal_smooth, 25)  # bottom 25%
        min_gap_ms = 8000
    
    # Scan for instrumental gaps
    is_instrumental = vocal_smooth < vocal_thresh
    knots = []
    
    i = 0
    while i < len(is_instrumental):
        if is_instrumental[i]:
            start_i = i
            while i < len(is_instrumental) and is_instrumental[i]:
                i += 1
            end_i = i - 1
            
            start_ms = float(spec_times_ms[start_i])
            end_ms = float(spec_times_ms[min(end_i, len(spec_times_ms) - 1)])
            gap_dur = end_ms - start_ms
            
            # Interior check — don't duplicate structural layer
            is_interior = start_ms > duration_ms * 0.08 and end_ms < duration_ms * 0.92
            
            if gap_dur >= min_gap_ms and is_interior:
                knots.append({
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "reason": f"instrumental gap ({gap_dur/1000:.1f}s)",
                    "layer": "vocal_gap",
                })
        i += 1
    
    return knots


# ══════════════════════════════════════════════════════════════
# LAYER 4: REPETITION — skip redundant repeated sections
# ══════════════════════════════════════════════════════════════
def _layer_repetition(chroma: np.ndarray, sr: int, sections: list,
                       sensitivity: str) -> list:
    """
    Detect sections that are near-duplicates of earlier sections
    using Chromagram + Self-Similarity Matrix (SSM).
    """
    if sensitivity == "light":
        return []  # light mode doesn't skip repeats
    
    from scipy.spatial.distance import cdist
    from scipy.ndimage import uniform_filter1d
    
    hop_length = 512
    knots = []
    
    # Only consider chorus/verse sections for repetition
    labeled = [(i, s) for i, s in enumerate(sections) if s["label"] in ("chorus", "verse")]
    
    if len(labeled) < 3:  # Need at least 3 to skip the 3rd+
        return []
    
    print("   Analyzing chromagram for repetition detection...")
    
    def _section_chroma_fingerprint(section):
        """Extract averaged chroma vector for a section."""
        start_frame = int(section["start_ms"] / 1000.0 * sr / hop_length)
        end_frame = int(section["end_ms"] / 1000.0 * sr / hop_length)
        start_frame = max(0, min(start_frame, chroma.shape[1] - 1))
        end_frame = max(start_frame + 1, min(end_frame, chroma.shape[1]))
        
        section_chroma = chroma[:, start_frame:end_frame]
        if section_chroma.shape[1] < 5:
            return None
        
        # Average the chroma across time to get a 12-dim "harmonic fingerprint"
        avg_chroma = np.mean(section_chroma, axis=1)
        # Also keep the chroma sequence for detailed comparison
        return avg_chroma, section_chroma
    
    # Build fingerprints
    fingerprints = []
    for idx, s in labeled:
        fp = _section_chroma_fingerprint(s)
        if fp is not None:
            fingerprints.append((idx, s, fp[0], fp[1]))
    
    if len(fingerprints) < 3:
        return []
    
    # Build pairwise cosine similarity matrix on averaged chroma
    avg_chromas = np.array([fp[2] for fp in fingerprints])  # (N, 12)
    norms = np.linalg.norm(avg_chromas, axis=1, keepdims=True) + 1e-10
    avg_chromas_normed = avg_chromas / norms
    sim_matrix = 1 - cdist(avg_chromas_normed, avg_chromas_normed, metric='cosine')
    
    # Group sections into clusters of similar harmonic content
    similarity_thresh = 0.80 if sensitivity == "aggressive" else 0.85
    
    # Find groups: each group is a list of section indices that sound similar
    groups = []  # list of lists of fingerprint indices
    assigned = set()
    
    for i in range(len(fingerprints)):
        if i in assigned:
            continue
        group = [i]
        assigned.add(i)
        for j in range(i + 1, len(fingerprints)):
            if j in assigned:
                continue
            if sim_matrix[i, j] >= similarity_thresh:
                # Same label check
                if fingerprints[i][1]["label"] == fingerprints[j][1]["label"]:
                    group.append(j)
                    assigned.add(j)
        groups.append(group)
    
    # For each group with 3+ members, skip the 3rd+ occurrence
    for group in groups:
        if len(group) < 3:
            continue
        
        # Sort by time (they should already be, but be safe)
        group_sorted = sorted(group, key=lambda g: fingerprints[g][1]["start_ms"])
        
        # Keep first 2 occurrences, knot the 3rd+
        for g_idx in group_sorted[2:]:
            sec = fingerprints[g_idx][1]
            dur_ms = sec["end_ms"] - sec["start_ms"]
            
            min_dur = 8000 if sensitivity == "balanced" else 5000
            if dur_ms >= min_dur:
                # Find similarity to the first occurrence
                first_sim = sim_matrix[group_sorted[0], g_idx]
                knots.append({
                    "start_ms": sec["start_ms"],
                    "end_ms": sec["end_ms"],
                    "reason": f"repeated {sec['label']} #{group_sorted.index(g_idx)+1} "
                              f"({dur_ms/1000:.1f}s, {first_sim:.0%} similar)",
                    "layer": "repetition",
                })
    
    return knots


# ══════════════════════════════════════════════════════════════
# MERGE + GUARDRAILS
# ══════════════════════════════════════════════════════════════
def merge_and_deduplicate(all_knots: list, duration_ms: float,
                           min_play_between_ms: float = 3000.0) -> list:
    """
    Merge overlapping/adjacent knot candidates from all layers.
    Apply guardrails:
      - Minimum 3s of audible playback between knots
      - Don't knot more than 60% of total duration
      - Merge knots with < 2s gap between them
    """
    if not all_knots:
        return []
    
    # Sort by start time
    sorted_knots = sorted(all_knots, key=lambda k: k["start_ms"])
    
    # Merge overlapping regions
    merged = [sorted_knots[0].copy()]
    for knot in sorted_knots[1:]:
        prev = merged[-1]
        # If overlapping or gap < 2s, merge
        if knot["start_ms"] <= prev["end_ms"] + 2000:
            prev["end_ms"] = max(prev["end_ms"], knot["end_ms"])
            prev["reason"] += f" + {knot['reason']}"
            # Keep the most significant layer
            if knot["layer"] == "structural":
                prev["layer"] = "structural"
        else:
            merged.append(knot.copy())
    
    # Guardrail: Don't knot more than 55% of total duration
    total_knotted = sum(k["end_ms"] - k["start_ms"] for k in merged)
    if total_knotted > duration_ms * 0.55:
        # Keep only the highest-confidence knots (shortest = most confident)
        # Sort by duration and drop the longest until under budget
        by_priority = sorted(merged, key=lambda k: {
            "structural": 0, "energy_valley": 1, "vocal_gap": 2, "repetition": 3
        }.get(k["layer"], 4))
        
        kept = []
        budget = duration_ms * 0.55
        running = 0
        for k in by_priority:
            dur = k["end_ms"] - k["start_ms"]
            if running + dur <= budget:
                kept.append(k)
                running += dur
        merged = sorted(kept, key=lambda k: k["start_ms"])
    
    return merged


# ══════════════════════════════════════════════════════════════
# LAYER 5 (PRO ENGINE): DEEP LEARNING VOCAL EXTRACTION
# ══════════════════════════════════════════════════════════════
def _layer_demucs_vocals(file_path: str, duration_ms: float) -> list:
    """
    Uses Demucs to separate vocals from the track.
    Analyzes the vocal stem and creates knots where vocal energy is absent.
    """
    import tempfile
    
    # Run demucs via subprocess
    print("🧠 [PRO ENGINE] Running Demucs deep vocal separation (this may take a few minutes)...")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(file_path)), "separated")
    
    cmd = [
        "/usr/local/bin/python3.11", "-m", "demucs",
        "--two-stems=vocals",
        "-o", out_dir,
        file_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"   ⚠️ Demucs failed: {e}. Falling back to fast analysis.")
        return []
        
    # The output is in separated/htdemucs/<filename_without_ext>/vocals.wav
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    vocal_path = os.path.join(out_dir, "htdemucs", base_name, "vocals.wav")
    
    if not os.path.exists(vocal_path):
        print("   ⚠️ Could not find extracted vocal stem.")
        return []
        
    print("   🎤 Analyzing extracted vocal stem...")
    y_voc, sr_voc = load_audio(vocal_path)
    energy_voc = compute_energy_curve(y_voc, sr_voc)
    
    rms = energy_voc["rms"]
    times_ms = energy_voc["times_ms"]
    
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(rms, size=30)
    
    # Vocal threshold: what counts as "no singing"
    peak_vocal = np.percentile(smoothed, 95)
    vocal_thresh = peak_vocal * 0.15  # less than 15% of peak vocal energy is silence
    
    below = smoothed < vocal_thresh
    knots = []
    
    min_gap_ms = 4000  # 4s minimum vocal gap
    
    i = 0
    while i < len(below):
        if below[i]:
            start_i = i
            while i < len(below) and below[i]:
                i += 1
            end_i = i - 1
            
            start_ms = float(times_ms[start_i])
            end_ms = float(times_ms[min(end_i, len(times_ms) - 1)])
            gap_dur = end_ms - start_ms
            
            # Don't knot the very beginning/end (let structural layer handle it)
            is_interior = start_ms > duration_ms * 0.05 and end_ms < duration_ms * 0.95
            
            if gap_dur >= min_gap_ms and is_interior:
                knots.append({
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "reason": f"pure instrumental gap ({gap_dur/1000:.1f}s)",
                    "layer": "demucs_vocal",
                })
        i += 1
        
    # Cleanup separated folder to save space
    try:
        shutil.rmtree(out_dir)
    except:
        pass
        
    return knots

# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
def run_analysis(file_path: str, sensitivity: str = "balanced", device_uri: str = None, engine: str = "fast") -> dict:
    """Run the full multi-layer analysis pipeline."""
    print(f"🎵 Loading audio: {os.path.basename(file_path)}")
    y, sr = load_audio(file_path)
    duration_ms = get_song_duration_ms(y, sr)
    print(f"   Duration: {duration_ms/1000:.1f}s | Sample rate: {sr}Hz | Samples: {len(y):,}")
    
    print("🧠 Extracting spectral features in streaming chunks (memory-efficient)...")
    feats = extract_features_chunked(y, sr)

    # ── Step 1: Energy curve ──
    print("📊 Computing energy curve...")
    energy = compute_energy_curve(y, sr)
    print(f"   Mean energy: {energy['mean_energy']:.4f} | Threshold: {energy['energy_threshold']:.4f}")

    # ── Step 2: Beat tracking ──
    print("🥁 Detecting beats...")
    beats = find_beats(feats["onset_env"], sr, len(y))
    print(f"   Tempo: {beats['tempo']:.1f} BPM | Beats: {len(beats['beat_times_ms'])} | "
          f"Downbeats: {len(beats['downbeat_times_ms'])}")

    # ── Step 3: Section detection ──
    print("🔍 Detecting song sections...")
    sections = detect_sections(feats["mfcc"], feats["chroma"], energy, sr, n_sections=10)
    sections = classify_sections(sections, energy["energy_threshold"])
    print(f"   Found {len(sections)} sections:")
    for i, s in enumerate(sections):
        dur = (s['end_ms'] - s['start_ms']) / 1000
        print(f"   [{i+1}] {s['label']:>8s} | {s['start_ms']/1000:6.1f}s → "
              f"{s['end_ms']/1000:6.1f}s ({dur:.1f}s) | energy: {s['energy']:.4f}")

    # ══════════════════════════════════════════════════════
    # MULTI-LAYER KNOT DETECTION
    # ══════════════════════════════════════════════════════
    all_knots = []

    # Layer 1: Structural
    print(f"\n🏗️  Layer 1: Structural analysis...")
    l1 = _layer_structural(sections, sensitivity)
    print(f"   → {len(l1)} candidates")
    all_knots.extend(l1)

    # Layer 2: Energy valleys
    print(f"📉 Layer 2: Energy valley detection...")
    l2 = _layer_energy_valleys(energy, duration_ms, sensitivity)
    print(f"   → {len(l2)} candidates")
    for k in l2:
        print(f"      {k['start_ms']/1000:.1f}s → {k['end_ms']/1000:.1f}s — {k['reason']}")
    all_knots.extend(l2)

    # Layer 3: Vocal gaps
    print(f"🎤 Layer 3: Vocal gap detection...")
    l3 = _layer_vocal_gaps(feats["centroids"], feats["flatness"], feats["spec_times_ms"], energy, duration_ms, sensitivity)
    print(f"   → {len(l3)} candidates")
    for k in l3:
        print(f"      {k['start_ms']/1000:.1f}s → {k['end_ms']/1000:.1f}s — {k['reason']}")
    all_knots.extend(l3)

    # Layer 4: Repetition
    print(f"🔁 Layer 4: Repetition detection...")
    l4 = _layer_repetition(feats["chroma"], sr, sections, sensitivity)
    print(f"   → {len(l4)} candidates")
    for k in l4:
        print(f"      {k['start_ms']/1000:.1f}s → {k['end_ms']/1000:.1f}s — {k['reason']}")
    all_knots.extend(l4)

    l5 = []
    if engine == "pro":
        l5 = _layer_demucs_vocals(file_path, duration_ms)
        print(f"   → {len(l5)} candidates")
        for k in l5:
            print(f"      {k['start_ms']/1000:.1f}s → {k['end_ms']/1000:.1f}s — {k['reason']}")
        all_knots.extend(l5)

    # ── Merge & Deduplicate ──
    print(f"\n🔀 Merging {len(all_knots)} total candidates...")
    knots = merge_and_deduplicate(all_knots, duration_ms)
    print(f"   → {len(knots)} final knot regions:")
    for i, k in enumerate(knots):
        dur = (k['end_ms'] - k['start_ms']) / 1000
        print(f"   [{i+1}] {k['start_ms']/1000:6.1f}s → {k['end_ms']/1000:6.1f}s "
              f"({dur:.1f}s) — {k['reason']}")

    # ── DSP refinement ──
    print("\n✂️  Refining knot boundaries (zero-crossing + beat snap)...")
    junctions = []
    for k in knots:
        refined_start, refined_end = refine_knot_boundary(
            y, sr, k["start_ms"], k["end_ms"],
            beats["beat_times_ms"], beats["downbeat_times_ms"]
        )
        confidence = validate_transition(y, sr, refined_start, refined_end)

        junctions.append({
            "start_ms": round(refined_start, 1),
            "end_ms": round(refined_end, 1),
        })
        print(f"   {k['start_ms']/1000:.1f}s → {refined_start/1000:.3f}s | "
              f"{k['end_ms']/1000:.1f}s → {refined_end/1000:.3f}s | "
              f"confidence: {confidence:.3f}")

    # ── Build output ──
    if device_uri is None:
        filename = os.path.basename(file_path)
        device_uri = f"file:///storage/emulated/0/Audior/{filename}"

    knotted_ms = sum(j["end_ms"] - j["start_ms"] for j in junctions)

    result = {
        "_id": device_uri,
        "name": "Auto-Knot",
        "junctions": junctions,
        "knotted_duration_ms": round(knotted_ms, 1),
        "original_duration_ms": round(duration_ms, 1),
        "createdAt": int(time.time() * 1000),
        "_meta": {
            "sensitivity": sensitivity,
            "layers_used": {
                "structural": len(l1),
                "energy_valley": len(l2),
                "vocal_gap": len(l3),
                "repetition": len(l4),
                "demucs_vocal": len(l5) if engine == "pro" else 0,
            },
            "sections": [
                {
                    "label": s["label"],
                    "start_ms": round(s["start_ms"], 1),
                    "end_ms": round(s["end_ms"], 1),
                    "energy": round(s["energy"], 4),
                }
                for s in sections
            ],
            "tempo_bpm": round(beats["tempo"], 1),
            "analysis_file": os.path.basename(file_path),
        },
    }

    # ── Summary ──
    play_ms = duration_ms - knotted_ms
    print(f"\n📋 Summary:")
    print(f"   Original:  {duration_ms/1000:.1f}s")
    print(f"   Knotted:   {knotted_ms/1000:.1f}s skipped")
    print(f"   Playback:  {play_ms/1000:.1f}s ({play_ms/duration_ms*100:.0f}% of original)")
    print(f"   Junctions: {len(junctions)}")
    if engine == "pro":
        print(f"   Layers:    L1={len(l1)} L2={len(l2)} L3={len(l3)} L4={len(l4)} L5(Pro)={len(l5)}")
    else:
        print(f"   Layers:    L1={len(l1)} L2={len(l2)} L3={len(l3)} L4={len(l4)}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Auto-Knotting Engine v2 — Deep multi-layer analysis")
    parser.add_argument("--file", required=True, help="Path to audio file")
    parser.add_argument("--sensitivity", default="balanced", choices=["aggressive", "balanced", "light"],
                        help="Knotting sensitivity (default: balanced)")
    parser.add_argument("--engine", default="fast", choices=["fast", "pro"],
                        help="Engine mode: fast (DSP only) or pro (AI Vocal Separation)")
    parser.add_argument("--output", default="output/result.json", help="Output JSON path")
    parser.add_argument("--device-uri", default=None,
                        help="Override the file:/// URI on the device (for _id field)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"❌ File not found: {args.file}")
        sys.exit(1)

    print("=" * 60)
    print(f"  🪢  AUTO-KNOTTING ENGINE v2 — {args.engine.upper()} MODE")
    print("=" * 60)
    print()

    result = run_analysis(args.file, args.sensitivity, args.device_uri, args.engine)

    # Save output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Saved to: {args.output}")

    # Print app-compatible JSON
    app_json = {k: v for k, v in result.items() if not k.startswith("_meta")}
    print(f"\n📱 App-compatible JSON:")
    print(json.dumps(app_json, indent=2))


if __name__ == "__main__":
    main()
