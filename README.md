# Auto-Knotting Engine

The Auto-Knotting Engine is a three-tier system designed to intelligently analyze audio tracks (like music) and determine optimal "knots" (skip points/junctions) based on structural, energetic, and harmonic properties of the sound. This ensures a seamless, engaging listening experience by automatically skipping dull or repetitive sections, while protecting important musical build-ups and snapping cuts to downbeats for natural transitions.

## Architecture: The Three Tiers

1. **Tier 1: Instant (In-App)**
   * **Location:** Runs locally on the user's device (JavaScript).
   * **Features:** Lightweight heuristics for detecting intros, outros, and long silent gaps.
   * **Pros/Cons:** Zero latency, works offline, but lacks deep musical understanding.

2. **Tier 2: Fast (Cloud DSP)**
   * **Location:** Deployed on Render (Python Flask server).
   * **Features:** Robust 4-layer DSP pipeline using `numpy`, `scipy`, and `librosa`/`mir` modules.
     * *Layer 1:* Structure (Intro/Outro trimming).
     * *Layer 2:* Energy Valleys (with build-up/crescendo protection).
     * *Layer 3:* Vocal Gap detection (via harmonic/percussive separation).
     * *Layer 4:* Repetition skipping (Chromagram + Self-Similarity Matrix to detect and skip 3rd+ chorus repetitions).
     * *Refinement:* Downbeat-priority snapping + zero-crossing alignment to ensure cuts fall on the "1" of the bar without audio clicks.
   * **Pros/Cons:** Fast processing (~30s), good musicality, but limited by traditional DSP methods.

3. **Tier 3: Pro (GPU AI)**
   * **Location:** Serverless deployment on Modal.com (T4 GPU).
   * **Features:** State-of-the-art AI pipeline.
     * Incorporates all Tier 2 features.
     * Uses **Demucs** for high-fidelity vocal/instrumental stem separation.
     * Uses **Whisper** for precise phrase-level vocal detection.
   * **Pros/Cons:** Unparalleled accuracy, near-perfect vocal gap skipping, but takes longer (~60-120s) and requires internet.

## Repository Structure

```
├── src/
│   ├── analyze.py       # Core orchestrator for the 4/5-layer pipeline
│   ├── dsp.py           # Digital Signal Processing utilities (beat snapping, filtering)
│   ├── mir.py           # Music Information Retrieval (chroma, onset detection)
│   ├── server.py        # Flask API for the Fast (Render) tier
│   ├── modal_pro.py     # Modal.com serverless definition for the Pro tier
│   ├── inject.py        # ID3 tag injection for knot metadata
│   └── visualize.py     # Debugging visualization tools
├── Dockerfile           # Minimal Python 3.11 image for Render
├── render.yaml          # Render deployment specification
├── requirements.txt     # Full dependency list (including ML models)
└── requirements-render.txt # Lightweight dependencies for Fast tier
```

## Deployment

### Fast Engine (Render)
1. Push this repository to GitHub.
2. Connect the repository in the Render Dashboard.
3. Use the `render.yaml` blueprint or configure a Docker web service using `Dockerfile`.
4. Ensure the plan accommodates the 150MB slug limit.

### Pro Engine (Modal)
1. Install Modal CLI: `pip install modal`
2. Authenticate: `modal token set`
3. Deploy the function: `modal deploy src/modal_pro.py`
4. The backend will forward base64 encoded audio to the generated Modal web endpoint.
