# Auto-Knotting Engine Workflow

The Auto-Knotting Engine is a multi-layer audio analysis pipeline that automatically identifies the most skippable or redundant parts of a song and creates "knots" (skipped sections) to keep the listener engaged. 

The engine currently supports two operating modes:
- **Fast Mode (`--engine fast`)**: Uses pure Digital Signal Processing (DSP) techniques. It is highly efficient and relies on spectral and energy features to find instrumentals, repetitions, and lulls.
- **Pro Mode (`--engine pro`)**: Adds an advanced AI layer using Facebook's Demucs deep learning model. It isolates the vocal stem to surgically find gaps where no vocals are present.

---

## 1. The Core Analysis Pipeline

The engine evaluates a song through several independent "layers." Each layer analyzes the track for specific musical patterns and proposes candidate knots. 

### Layer 1: Structural Analysis
Identifies high-level song structures.
- Detects the **intro**, **outro**, and **bridge**.
- Knots excessively long intros/outros and low-energy bridges depending on the chosen sensitivity (Aggressive, Balanced, or Light).

### Layer 2: Energy Valleys
Finds sustained quiet dips in the middle of the song.
- Calculates a smoothed Root Mean Square (RMS) energy curve.
- Proposes knots for contiguous regions where energy drops significantly below the song's average (e.g., musical pauses, quiet build-ups) for at least a few seconds.

### Layer 3: Vocal Gap Detection (Spectral Analysis)
A purely DSP-based attempt to find instrumental passages.
- Computes the **Spectral Centroid** and **Spectral Flatness**. 
- Human vocals typically have a higher centroid (1000-4000 Hz) and varying flatness compared to smooth, tonal instrumental pads.
- Proposes knots for long, continuous passages that score very low on the "vocal activity" likelihood.

### Layer 4: Repetition (Chorus/Verse Duplicates)
Skips redundant parts of the song to keep the flow moving.
- Extracts short feature snippets (using RMS envelope) for sections labeled "chorus" or "verse".
- Uses **Pearson cross-correlation** to compare sections.
- If a later chorus or verse is >80% similar to an earlier one, it is proposed as a knot (so the listener only hears it once or twice).

### Layer 5: AI Vocal Separation (PRO Engine Only)
The most accurate method for finding true instrumental sections.
- Runs **Demucs** in a subprocess to split the track into a vocal stem and an instrumental stem.
- Analyzes *only* the vocal stem.
- Proposes knots for any section >4 seconds where vocal energy drops to near-zero (i.e., pure instrumentals).

---

## 2. Merging & Guardrails

Once all layers have proposed their candidate knots, the engine merges them together to ensure the song remains listenable.

- **Merging**: Overlapping knots from different layers are combined.
- **Proximity Guardrail**: Knots that are less than 2 seconds apart are merged into one large knot, preventing annoying "stuttering" playbacks.
- **Budget Guardrail**: The engine enforces a strict rule: **No more than 55% of the song's duration can be knotted.** If the budget is exceeded, it drops the lowest-priority knots, keeping the most structural and AI-confirmed knots first.

---

## 3. DSP Refinement (The Snap)

After the final list of knots is approved, the engine fine-tunes the start and end timestamps so the audio transitions smoothly.

- **Beat Snapping**: Align knot boundaries with the nearest detected beat or downbeat so the rhythm doesn't break abruptly.
- **Zero-Crossing**: Snaps the final millisecond to a point where the audio waveform crosses the zero-axis, preventing a "popping" or "clicking" sound when the player jumps across the knot.

---

## 4. Final Output

The engine spits out a JSON result compatible with the Knot mobile app containing:
1. `_id`: The device URI mapping.
2. `junctions`: The exact `start_ms` and `end_ms` for every knot.
3. `knotted_duration_ms`: Total time skipped.
4. `_meta`: Developer metadata tracking which layer contributed which knots, tempo, and total sections.

This JSON is injected into the device's Upstash Redis backend via the `inject.py` script.
