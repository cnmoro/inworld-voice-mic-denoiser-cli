#!/usr/bin/env python3
"""Checks the CLI against the upstream golden fixtures (tolerance 1e-4).

The fixtures are downloaded from github.com/inworld-ai/voice-isolator at test
time (CC BY 4.0 audio, not vendored here).

usage: golden.py <path to inworldvoice> [--cpu|--gpu]
"""
import subprocess, sys, tempfile, urllib.request, os
import numpy as np

BASE = "https://raw.githubusercontent.com/inworld-ai/voice-isolator/main/tests/fixtures"
exe, flag = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "--cpu")
failed = False
with tempfile.TemporaryDirectory() as tmp:
    for name in ("offline_3s", "offline_11s"):
        wav, npy = os.path.join(tmp, name + ".wav"), os.path.join(tmp, name + ".npy")
        urllib.request.urlretrieve(f"{BASE}/{name}/input.wav", wav)
        urllib.request.urlretrieve(f"{BASE}/{name}/expected.npy", npy)
        out = os.path.join(tmp, name + "_out.wav")
        subprocess.run([exe, flag, "-o", out, wav], check=True)
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", out, "-f", "f32le", "-ar", "48000", "-ac", "1", "-"],
                             check=True, capture_output=True).stdout
        got, want = np.frombuffer(raw, dtype="<f4"), np.load(npy).astype("float32").reshape(-1)
        n = min(len(got), len(want))
        err = float(np.abs(got[:n] - want[:n]).max())
        # the output file is 16-bit PCM like the input: allow its quantisation step
        ok = len(got) == len(want) and err <= 1e-4 + 3.1e-5
        print(f"{name}: samples {len(got)}/{len(want)} max abs error {err:.2e} {'OK' if ok else 'FAIL'}")
        failed |= not ok
sys.exit(1 if failed else 0)
