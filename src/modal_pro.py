#!/usr/bin/env python3
"""
modal_pro.py — Pro Auto-Knotting Engine on Modal.com (GPU Serverless)
======================================================================
Runs the full AI pipeline with Demucs vocal separation + Whisper
phrase detection on a T4 GPU via Modal.com's serverless infrastructure.

Deployment:
  modal deploy src/modal_pro.py

Usage:
  curl -X POST https://YOUR_APP--knot-pro-analyze.modal.run \
    -F "file=@song.m4a"

Prerequisites:
  - pip install modal
  - modal token set --token-id <YOUR_TOKEN_ID> --token-secret <YOUR_TOKEN_SECRET>
  
Token Placeholder:
  Set your Modal token via:
    modal token set --token-id YOUR_ID --token-secret YOUR_SECRET
  Or via environment variables:
    MODAL_TOKEN_ID=xxx
    MODAL_TOKEN_SECRET=xxx
"""

import modal

# ── Modal App Definition ──
app = modal.App("knot-pro")

# ── Container Image ──
# Pre-install all heavy dependencies in the image so cold starts are fast
pro_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "soundfile>=0.12.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "demucs>=4.0.0",
        "faster-whisper>=1.0.0",
        "flask>=3.0.0",
    )
)


# -- Source Mount --
src_mount = modal.Mount.from_local_dir(
    "src",
    remote_path="/root"
)

@app.function(
    image=pro_image,
    gpu="T4",
    timeout=600,  # Increased to 10 minutes for Demucs
    memory=8192,  # Increased to 8GB for Demucs
    mounts=[src_mount],
    allow_concurrent_inputs=5,
)
def analyze_pro(audio_bytes: bytes, filename: str = "input.m4a",
                sensitivity: str = "balanced") -> dict:
    """
    Full Pro analysis pipeline:
    1. Save audio to temp file
    2. Run Demucs vocal separation (GPU accelerated)
    3. Run standard 4-layer DSP analysis
    4. Run Demucs vocal gap layer (Layer 5)
    5. Optionally run Whisper for phrase-level timestamps
    6. Merge, deduplicate, and refine
    """
    import os
    import sys
    import time
    import tempfile
    
    start_time = time.time()
    
    # Save audio bytes to temp file
    suffix = os.path.splitext(filename)[1] or '.m4a'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    try:
        # Add source directory to path
        sys.path.insert(0, '/root')
        
        from analyze import run_analysis
        
        result = run_analysis(
            file_path=tmp_path,
            sensitivity=sensitivity,
            engine='pro',  # Enables Demucs Layer 5
        )
        
        elapsed = time.time() - start_time
        
        return {
            'junctions': result['junctions'],
            'knotted_duration_ms': result['knotted_duration_ms'],
            'original_duration_ms': result['original_duration_ms'],
            'analysis_time_s': round(elapsed, 2),
            'gpu': 'T4',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.function(
    image=pro_image,
    timeout=10,
)
def health_check() -> dict:
    """Health check endpoint."""
    import time
    return {
        'status': 'ok',
        'engine': 'pro',
        'gpu': 'T4',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
    }


# ── Web Endpoint ──
# This creates a public HTTPS endpoint that the Node.js backend can call
@app.function(
    image=pro_image,
    timeout=600,
)
@modal.web_endpoint(method="POST")
async def analyze_web(request: modal.Request) -> dict:
    """
    Web endpoint for the Pro engine.
    
    Accepts multipart/form-data:
    - file: Audio file
    - sensitivity: 'balanced' | 'aggressive' | 'light'
    - engine: 'pro'
    """
    form = await request.form()
    audio_file = form.get('file')
    sensitivity = form.get('sensitivity', 'balanced')
    
    if not audio_file:
        return {'error': 'No file in request body'}
    
    # Read the file bytes
    audio_bytes = await audio_file.read()
    
    # Trigger the heavy GPU function remotely
    result = analyze_pro.remote(
        audio_bytes=audio_bytes,
        filename=audio_file.filename or 'input.m4a',
        sensitivity=sensitivity,
    )
    
    return result


# ── Local Source Files Mount ──
# Copy the engine source files into the container
@app.cls(
    image=pro_image.copy_local_dir("src", "/root"),
    gpu="T4",
    timeout=300,
    memory=4096,
)
class ProEngine:
    """Class-based interface for the Pro engine with persistent state."""
    
    @modal.method()
    def analyze(self, audio_bytes: bytes, filename: str = "input.m4a",
                sensitivity: str = "balanced") -> dict:
        return analyze_pro.local(
            audio_bytes=audio_bytes,
            filename=filename,
            sensitivity=sensitivity,
        )
    
    @modal.method()
    def health(self) -> dict:
        return health_check.local()


# ── CLI Entry Point ──
@app.local_entrypoint()
def main():
    """Test the Pro engine locally."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: modal run src/modal_pro.py -- <audio_file>")
        return
    
    audio_path = sys.argv[1]
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()
    
    print(f"Sending {len(audio_bytes)} bytes to Pro engine...")
    result = analyze_pro.remote(
        audio_bytes=audio_bytes,
        filename=audio_path,
    )
    
    import json
    print(json.dumps(result, indent=2))
