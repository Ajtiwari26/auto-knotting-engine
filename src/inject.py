#!/usr/bin/env python3
"""
inject.py — ADB AsyncStorage Injection
=========================================
Injects auto-knot results directly into the app's AsyncStorage
SQLite database on the Android device via ADB.

Requires:
  - Debug build of the app installed on device (run-as must work)
  - ADB connected to device

Usage:
  python3 src/inject.py \\
    --result output/result.json \\
    --device 10BF5P2AZF0010T \\
    --package com.ajay.knot
"""

import argparse
import json
import os
import subprocess
import sys


def adb_cmd(args: list, device: str = None) -> str:
    """Run an ADB command and return stdout."""
    cmd = ["adb"]
    if device:
        cmd.extend(["-s", device])
    cmd.extend(args)
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ADB command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout.strip()


def verify_debug_build(device: str, package: str):
    """Check that the app is a debug build (run-as must work)."""
    try:
        output = adb_cmd(["shell", f"run-as {package} ls databases/RKStorage"], device)
        if "RKStorage" not in output and "No such file" in output:
            raise RuntimeError("RKStorage database not found")
        return True
    except RuntimeError as e:
        if "not debuggable" in str(e):
            print("❌ App is not a debug build. Install a debug build first:")
            print("   cd mobile && npx expo run:android")
            sys.exit(1)
        raise


def read_existing_knots(device: str, package: str) -> list:
    """Read all existing knot keys from AsyncStorage."""
    try:
        output = adb_cmd([
            "shell",
            f"run-as {package} sqlite3 databases/RKStorage "
            f"\"SELECT key FROM catalystLocalStorage WHERE key LIKE 'knot_data_%';\""
        ], device)
        return [line.strip() for line in output.split('\n') if line.strip()]
    except RuntimeError:
        return []


def inject_knot(device: str, package: str, key: str, value_json: str):
    """
    Insert or replace a knot entry in AsyncStorage.
    
    Strategy: Write a SQL file to /data/local/tmp, copy it into the app's
    sandbox via run-as, then execute it with sqlite3 .read.
    This avoids shell escaping issues with JSON and URIs.
    """
    import tempfile

    # Escape single quotes in JSON for SQLite
    escaped_json = value_json.replace("'", "''")
    escaped_key = key.replace("'", "''")
    
    sql = f"INSERT OR REPLACE INTO catalystLocalStorage (key, value) VALUES ('{escaped_key}', '{escaped_json}');\n"
    
    # Write SQL to a local temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
        f.write(sql)
        local_sql_path = f.name
    
    try:
        # Push SQL file to device's accessible temp directory
        device_tmp = "/data/local/tmp/_knot_inject.sql"
        subprocess.run(
            ["adb", "-s", device, "push", local_sql_path, device_tmp],
            capture_output=True, check=True, timeout=10
        )
        
        # Copy the file into the app's sandbox using run-as + cat
        subprocess.run(
            ["adb", "-s", device, "shell",
             f"run-as {package} sh -c 'cat {device_tmp} > /data/local/tmp/_knot_inject_copy.sql'"],
            capture_output=True, timeout=10
        )
        
        # Execute the SQL file using sqlite3's .read command
        result = subprocess.run(
            ["adb", "-s", device, "shell",
             f"run-as {package} sqlite3 databases/RKStorage '.read {device_tmp}'"],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode != 0 and result.stderr.strip():
            raise RuntimeError(f"SQLite error: {result.stderr.strip()}")
        
        # Clean up device temp file
        subprocess.run(
            ["adb", "-s", device, "shell", f"rm -f {device_tmp}"],
            capture_output=True, timeout=5
        )
    finally:
        os.unlink(local_sql_path)


def verify_injection(device: str, package: str, key: str) -> bool:
    """Verify that the knot was successfully injected."""
    try:
        output = adb_cmd([
            "shell",
            f"run-as {package} sqlite3 databases/RKStorage "
            f"\"SELECT length(value) FROM catalystLocalStorage WHERE key='{key}';\""
        ], device)
        length = int(output.strip())
        return length > 0
    except (RuntimeError, ValueError):
        return False


def force_stop_app(device: str, package: str):
    """Force stop the app so it reloads AsyncStorage on next launch."""
    try:
        adb_cmd(["shell", f"am force-stop {package}"], device)
    except RuntimeError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Inject auto-knot result into device AsyncStorage")
    parser.add_argument("--result", required=True, help="Path to result.json from analyze.py")
    parser.add_argument("--device", default="10BF5P2AZF0010T", help="ADB device serial")
    parser.add_argument("--package", default="com.ajay.knot", help="App package name")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be injected without doing it")
    args = parser.parse_args()

    # Load result
    with open(args.result) as f:
        result = json.load(f)

    song_id = result["_id"]
    key = f"knot_data_{song_id}"

    # Build the app-compatible JSON (strip _meta)
    app_data = {k: v for k, v in result.items() if not k.startswith("_meta")}
    value_json = json.dumps(app_data, separators=(',', ':'))  # compact

    print("=" * 60)
    print("  💉 AUTO-KNOT INJECTION")
    print("=" * 60)
    print()
    print(f"📱 Device:  {args.device}")
    print(f"📦 Package: {args.package}")
    print(f"🔑 Key:     {key}")
    print(f"📐 Size:    {len(value_json)} bytes")
    print(f"🪢  Knots:   {len(result.get('junctions', []))}")
    print()

    # Show junctions
    for i, j in enumerate(result.get("junctions", [])):
        print(f"   [{i+1}] Skip {j['start_ms']/1000:.1f}s → {j['end_ms']/1000:.1f}s "
              f"({(j['end_ms']-j['start_ms'])/1000:.1f}s)")

    if args.dry_run:
        print(f"\n🔍 DRY RUN — JSON that would be injected:")
        print(json.dumps(app_data, indent=2))
        return

    # Verify debug build
    print(f"\n🔍 Verifying debug build...")
    verify_debug_build(args.device, args.package)
    print("   ✅ Debug build confirmed")

    # Check existing knots
    existing = read_existing_knots(args.device, args.package)
    if key in existing:
        print(f"   ⚠️  Knot already exists for this song — will be REPLACED")
    print(f"   📊 {len(existing)} existing knotted songs on device")

    # Inject
    print(f"\n💉 Injecting knot data...")
    inject_knot(args.device, args.package, key, value_json)

    # Verify
    if verify_injection(args.device, args.package, key):
        print("   ✅ Injection verified — data is in AsyncStorage!")
    else:
        print("   ❌ Injection verification failed!")
        sys.exit(1)

    # Force restart app
    print(f"\n🔄 Restarting app...")
    force_stop_app(args.device, args.package)
    print("   ✅ App force-stopped. Open the app manually.")

    print(f"\n" + "=" * 60)
    print(f"  ✅ DONE!")
    print(f"  Open the Knot app → Knotted Library")
    print(f"  '{os.path.basename(song_id)}' should appear as a knotted song.")
    print(f"  Play it to verify the auto-knots!")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
