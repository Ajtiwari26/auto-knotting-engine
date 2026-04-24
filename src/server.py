#!/usr/bin/env python3
"""
server.py — Auto-Knotting Engine API Server
=============================================
Flask/Gunicorn server for Render deployment (Fast tier).
Accepts audio file uploads and returns knot JSON.

Deployment:
  gunicorn src.server:app --bind 0.0.0.0:$PORT --timeout 120
"""

import os
import sys
import time
import json
import tempfile
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze import run_analysis

app = Flask(__name__)
CORS(app)

# Max file size: 25MB for Fast tier
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'engine': 'fast',
        'version': '2.0',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze an uploaded audio file and return auto-knot results.
    
    Accepts:
      - multipart/form-data with 'file' field
      - JSON with 'song_uri', 'song_title', 'duration_ms'
    
    Returns:
      JSON with junctions, knotted_duration_ms, original_duration_ms
    """
    start_time = time.time()
    
    try:
        # Check if it's a file upload
        if 'file' in request.files:
            audio_file = request.files['file']
            if audio_file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            sensitivity = request.form.get('sensitivity', 'balanced')
            device_uri = request.form.get('device_uri', None)
            
            # Save to temp file
            suffix = os.path.splitext(audio_file.filename)[1] or '.m4a'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                audio_file.save(tmp.name)
                tmp_path = tmp.name
        
        elif request.is_json:
            data = request.get_json()
            # For JSON requests, the audio must be pre-uploaded or accessible
            # This is used when the backend proxies from the Node.js server
            if 'file_path' not in data:
                return jsonify({'error': 'No file_path in JSON body'}), 400
            
            tmp_path = data['file_path']
            sensitivity = data.get('sensitivity', 'balanced')
            device_uri = data.get('device_uri', None)
        
        else:
            return jsonify({'error': 'Send multipart/form-data with file or JSON with file_path'}), 400
        
        if not os.path.exists(tmp_path):
            return jsonify({'error': f'File not found: {tmp_path}'}), 404
        
        # Run analysis (Fast engine only on Render)
        print(f"[API] Analyzing: {tmp_path} (sensitivity={sensitivity})")
        result = run_analysis(
            file_path=tmp_path,
            sensitivity=sensitivity,
            device_uri=device_uri,
            engine='fast',
        )
        
        # Clean up temp file (only if we created it)
        if 'file' in request.files:
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        elapsed = time.time() - start_time
        
        # Return clean response (no _meta for external API)
        response = {
            'junctions': result['junctions'],
            'knotted_duration_ms': result['knotted_duration_ms'],
            'original_duration_ms': result['original_duration_ms'],
            'analysis_time_s': round(elapsed, 2),
            'engine': 'fast',
            'knot_count': len(result['junctions']),
        }
        
        # Include meta only if requested
        if request.args.get('include_meta') == 'true':
            response['_meta'] = result.get('_meta', {})
        
        print(f"[API] Done in {elapsed:.1f}s — {len(result['junctions'])} knots")
        return jsonify(response)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'analysis_time_s': round(time.time() - start_time, 2),
        }), 500


@app.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    """Analyze multiple files in sequence."""
    if not request.is_json:
        return jsonify({'error': 'JSON body required'}), 400
    
    data = request.get_json()
    files = data.get('files', [])
    results = []
    
    for f in files:
        try:
            result = run_analysis(
                file_path=f['file_path'],
                sensitivity=f.get('sensitivity', 'balanced'),
                device_uri=f.get('device_uri'),
                engine='fast',
            )
            results.append({
                'file': f['file_path'],
                'junctions': result['junctions'],
                'knotted_duration_ms': result['knotted_duration_ms'],
                'original_duration_ms': result['original_duration_ms'],
                'status': 'ok',
            })
        except Exception as e:
            results.append({
                'file': f['file_path'],
                'error': str(e),
                'status': 'error',
            })
    
    return jsonify({'results': results})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
