"""
server.py — Python API for Auto-Knotting
=========================================
Accepts audio file uploads and returns knot JSON.

Deployment:
# Note: Gunicorn timeout increased to 300s to match ffmpeg decoding needs for long songs
# Usage: gunicorn src.server:app --bind 0.0.0.0:$PORT --timeout 300
"""

import os
import tempfile
import json
from flask import Flask, request, jsonify
from src.analyze import analyze_song

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "knot-engine"}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    audio_file = request.files['file']
    sensitivity = request.form.get('sensitivity', 'balanced')
    device_uri = request.form.get('device_uri', '')
    
    # Create a temporary file to store the upload
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio_file.filename)[1], delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name
        
    try:
        print(f"--- Analysis Start: {audio_file.filename} ---")
        result = analyze_song(tmp_path, sensitivity=sensitivity)
        
        # Add metadata
        result["_id"] = device_uri or audio_file.filename
        
        return jsonify(result), 200
    
    except Exception as e:
        print(f"!!! Analysis Error: {str(e)} !!!")
        return jsonify({"error": f"Engine error: {str(e)}"}), 500
        
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
            print(f"--- Cleanup Complete ---")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
