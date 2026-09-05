#!/usr/bin/env python3
"""
Experimento de offsets OFFLINE sobre frame nativo salvo (sem dispositivo).
Usa diag_diamond/frame.png capturado quando o jogo estava visivel.
Nenhuma jogada, nenhuma alteracao no codigo do jogo.
"""

import sys
import os
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_diamond_trace import trace_slot, DXDY_LIMITS
from vision import _build_signal_map, read_pieces

BASE = os.path.dirname(os.path.abspath(__file__))
FRAME_PATH = os.path.join(BASE, "diag_diamond", "frame.png")


def main():
    fr = cv2.imread(FRAME_PATH)
    assert fr is not None, f"frame nao encontrado: {FRAME_PATH}"
    print(f"frame offline: {tuple(fr.shape)} (nativo esperado (2340,1080,3))", flush=True)
    assert fr.shape[0] == 2340 and fr.shape[1] == 1080, "frame nao esta em resolucao nativa!"

    sig = _build_signal_map(fr)
    print(f"signal: shape={tuple(sig.shape)} range=[{sig.min():.4f},{sig.max():.4f}]", flush=True)

    print("--- read_pieces() real (vision.py) ---", flush=True)
    for i, p in enumerate(read_pieces(fr)):
        if p and p.cells:
            print(f"slot {i}: {p.width}x{p.height} blocos={len(p.cells)} cells={sorted(p.cells)}", flush=True)
        else:
            print(f"slot {i}: None", flush=True)

    for lo, hi in DXDY_LIMITS:
        print(f"\n{'#'*70}\nEXPERIMENTO dx,dy em [{lo:+.0f},{hi:+.0f}] (passo 3px)\n{'#'*70}", flush=True)
        for slot in range(3):
            trace_slot(fr, sig, slot, dxdy_lim=(lo, hi))


if __name__ == "__main__":
    main()
