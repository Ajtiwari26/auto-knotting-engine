#!/usr/bin/env python3
"""
visualize.py — Debug Visualization
=====================================
Generates a PNG showing waveform, energy curve, section labels,
and knot regions for visual verification of the engine's output.

Usage:
  python3 src/visualize.py \\
    --file "test-samples/Pehle Kabhi Na Mera Haal..." \\
    --result output/result.json \\
    --output output/analysis.png
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mir import load_audio, compute_energy_curve


# Color scheme for section labels
SECTION_COLORS = {
    "intro": "#6366f1",    # indigo
    "verse": "#22c55e",    # green
    "chorus": "#ef4444",   # red (high energy)
    "bridge": "#f59e0b",   # amber
    "outro": "#8b5cf6",    # purple
    "unknown": "#94a3b8",  # gray
}


def generate_visualization(file_path: str, result: dict, output_path: str):
    """Generate a multi-panel visualization PNG."""
    
    print(f"📊 Loading audio for visualization...")
    y, sr = load_audio(file_path)
    duration_s = len(y) / sr
    times = np.linspace(0, duration_s, len(y))

    energy = compute_energy_curve(y, sr)

    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True,
                              gridspec_kw={'height_ratios': [2, 1.5, 0.5]})
    fig.patch.set_facecolor('#0f172a')  # dark background

    # ─── Panel 1: Waveform + Knot Regions ───
    ax1 = axes[0]
    ax1.set_facecolor('#1e293b')
    
    # Downsample waveform for plotting (every 100th sample)
    step = max(1, len(y) // 5000)
    ax1.plot(times[::step], y[::step], color='#38bdf8', linewidth=0.3, alpha=0.8)
    
    # Highlight knot regions (regions to SKIP) in red
    junctions = result.get("junctions", [])
    for j in junctions:
        start_s = j["start_ms"] / 1000.0
        end_s = j["end_ms"] / 1000.0
        ax1.axvspan(start_s, end_s, color='#ef4444', alpha=0.25, label='_nolegend_')
        ax1.axvline(start_s, color='#ef4444', linewidth=1, linestyle='--', alpha=0.7)
        ax1.axvline(end_s, color='#ef4444', linewidth=1, linestyle='--', alpha=0.7)

    ax1.set_ylabel('Amplitude', color='#e2e8f0', fontsize=11)
    ax1.set_title(f'🪢 Auto-Knot Analysis: {os.path.basename(file_path)}',
                  color='#f1f5f9', fontsize=14, fontweight='bold', pad=10)
    ax1.tick_params(colors='#94a3b8')

    # ─── Panel 2: Energy Curve + Threshold ───
    ax2 = axes[1]
    ax2.set_facecolor('#1e293b')
    
    energy_times_s = energy["times_ms"] / 1000.0
    ax2.fill_between(energy_times_s, energy["rms"], color='#38bdf8', alpha=0.4)
    ax2.plot(energy_times_s, energy["rms"], color='#38bdf8', linewidth=1)
    
    # Threshold line
    ax2.axhline(energy["energy_threshold"], color='#fbbf24', linewidth=1.5,
                linestyle='--', label=f'Threshold ({energy["energy_threshold"]:.4f})')
    
    # Knot regions on energy panel too
    for j in junctions:
        start_s = j["start_ms"] / 1000.0
        end_s = j["end_ms"] / 1000.0
        ax2.axvspan(start_s, end_s, color='#ef4444', alpha=0.15)

    ax2.set_ylabel('RMS Energy', color='#e2e8f0', fontsize=11)
    ax2.legend(loc='upper right', facecolor='#334155', edgecolor='#475569',
               labelcolor='#e2e8f0', fontsize=9)
    ax2.tick_params(colors='#94a3b8')

    # ─── Panel 3: Section Labels ───
    ax3 = axes[2]
    ax3.set_facecolor('#1e293b')

    meta = result.get("_meta", {})
    sections = meta.get("sections", [])

    for s in sections:
        start_s = s["start_ms"] / 1000.0
        end_s = s["end_ms"] / 1000.0
        color = SECTION_COLORS.get(s["label"], "#94a3b8")
        ax3.axvspan(start_s, end_s, color=color, alpha=0.7)
        # Label in the center of the section
        mid = (start_s + end_s) / 2.0
        ax3.text(mid, 0.5, s["label"].upper(), ha='center', va='center',
                 fontsize=8, fontweight='bold', color='white',
                 transform=ax3.get_xaxis_transform())

    ax3.set_ylabel('Sections', color='#e2e8f0', fontsize=11)
    ax3.set_xlabel('Time (seconds)', color='#e2e8f0', fontsize=11)
    ax3.set_yticks([])
    ax3.tick_params(colors='#94a3b8')

    # ─── Legend ───
    legend_patches = [
        mpatches.Patch(color='#ef4444', alpha=0.4, label='Knotted (skip)'),
    ]
    for label, color in SECTION_COLORS.items():
        if label != "unknown":
            legend_patches.append(mpatches.Patch(color=color, alpha=0.7, label=label.title()))

    fig.legend(handles=legend_patches, loc='lower center', ncol=6,
               facecolor='#334155', edgecolor='#475569', labelcolor='#e2e8f0',
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    # ─── Stats annotation ───
    knotted_ms = sum(j["end_ms"] - j["start_ms"] for j in junctions)
    original_ms = result.get("original_duration_ms", duration_s * 1000)
    play_pct = ((original_ms - knotted_ms) / original_ms * 100) if original_ms > 0 else 100

    stats_text = (
        f"Tempo: {meta.get('tempo_bpm', '?')} BPM | "
        f"Junctions: {len(junctions)} | "
        f"Skipped: {knotted_ms/1000:.1f}s | "
        f"Playback: {play_pct:.0f}%"
    )
    fig.text(0.5, 0.01, stats_text, ha='center', va='bottom',
             fontsize=10, color='#94a3b8', style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"✅ Visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate knot analysis visualization")
    parser.add_argument("--file", required=True, help="Path to audio file")
    parser.add_argument("--result", required=True, help="Path to result.json from analyze.py")
    parser.add_argument("--output", default="output/analysis.png", help="Output PNG path")
    args = parser.parse_args()

    with open(args.result) as f:
        result = json.load(f)

    generate_visualization(args.file, result, args.output)


if __name__ == "__main__":
    main()
