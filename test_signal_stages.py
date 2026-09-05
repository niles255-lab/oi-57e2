#!/usr/bin/env python3
"""
Disseccao do signal map por estagio (diagnostico apenas).
Compara: bruto vs boxFilter r=13 vs boxFilter menor vs sem/com adjacencia.
Nenhuma alteracao no codigo do jogo. Nenhuma jogada.
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from vision import NATIVE_DEVICE_W, NATIVE_DEVICE_H, _snap_to_catalog, _normalize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_signal")
os.makedirs(OUT, exist_ok=True)
FRAME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "diag_diamond", "frame.png")


def components(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    colored_f = ((s >= 80) & (v >= 45)).astype(np.float32)
    gray_f = ((s < 65) & (v >= 110)).astype(np.float32)
    mask_f = np.clip(colored_f + gray_f, 0.0, 1.0)
    color_f = (s >= 55).astype(np.float32)
    bright_f = ((s >= 35) & (v >= 50)).astype(np.float32)
    return mask_f, color_f, bright_f


def blur_map(mask_f, color_f, bright_f, half):
    k = 2 * half + 1
    sm = cv2.boxFilter(mask_f, -1, (k, k))
    sc = cv2.boxFilter(color_f, -1, (k, k))
    sb = cv2.boxFilter(bright_f, -1, (k, k))
    return sm * 0.55 + sc * 0.30 + sb * 0.15


def grid_points(sx, sy, pitch):
    cc = np.arange(7, dtype=np.float64) - 3.0
    rr = np.arange(7, dtype=np.float64) - 3.0
    xi = np.round(sx + cc * pitch).astype(np.int32)   # (7,)
    yi = np.round(sy + rr * pitch).astype(np.int32)   # (7,)
    return xi, yi


def eval_grid(sig, W, H, sx, sy, pitch, adj_thr=None, tag=""):
    """Avalia grade 7x7 fixa (pitch,dx=0,dy=0): bit por ponto + bbox + celulas."""
    xi, yi = grid_points(sx, sy, pitch)
    xo = (xi < 0) | (xi >= W)
    yo = (yi < 0) | (yi >= H)
    xic = np.clip(xi, 0, W - 1)
    yic = np.clip(yi, 0, H - 1)
    vals = sig[yic[:, np.newaxis], xic[np.newaxis, :]].astype(np.float64)
    oob = yo[:, np.newaxis] | xo[np.newaxis, :]
    vals = np.where(oob, 0.0, vals)
    strict = vals >= 0.60
    if adj_thr is None:
        bits = strict
    else:
        adj = np.zeros_like(strict)
        adj[:-1, :] |= strict[1:, :]
        adj[1:, :] |= strict[:-1, :]
        adj[:, :-1] |= strict[:, 1:]
        adj[:, 1:] |= strict[:, :-1]
        bits = strict | ((vals >= adj_thr) & adj)
    rw = np.where(bits.any(axis=1))[0]
    cw_ = np.where(bits.any(axis=0))[0]
    if rw.size == 0 or cw_.size == 0:
        print(f"  [{tag}] vazio", flush=True)
        return None
    r0, r1 = int(rw[0]), int(rw[-1]) + 1
    c0, c1 = int(cw_[0]), int(cw_[-1]) + 1
    bb = bits[r0:r1, c0:c1]
    det = [(r, c) for r in range(r1 - r0) for c in range(c1 - c0) if bb[r, c]]
    print(f"  [{tag}] n={len(det)} cells={sorted(_normalize(det)) if det else det}", flush=True)
    print(f"    grade 7x7 (S=strict . =adj/fundo, valor=signal):", flush=True)
    for r in range(7):
        row = []
        for c in range(7):
            ch = "#" if strict[r, c] else ("+" if bits[r, c] else ".")
            row.append(f"{ch}{vals[r, c]:.2f}")
        print(f"    r{r}: " + " ".join(row), flush=True)
    return det, vals, bits, (r0, r1, c0, c1)


EXPECTED = {
    0: [(0, 0), (0, 1), (0, 2)],
    1: [(0, 0), (0, 1), (1, 1)],
    2: [(0, 0), (1, 0), (2, 0)],
}


def norm(cells):
    if not cells:
        return []
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    mr, mc = min(rs), min(cs)
    return sorted((r - mr, c - mc) for r, c in cells)


def main():
    fr = cv2.imread(FRAME_PATH)
    assert fr is not None, FRAME_PATH
    H, W = fr.shape[:2]
    print(f"frame: {W}x{H}", flush=True)
    sc = W / NATIVE_DEVICE_W
    pitch = 58.0 * sc

    mask_f, color_f, bright_f = components(fr)
    print(f"mask_f>0: {mask_f.mean():.4f}", flush=True)
    cv2.imwrite(os.path.join(OUT, "mask_f.png"), (mask_f * 255).astype(np.uint8))

    maps = {}
    for r in (5, 6, 7, 8, 9, 13):
        maps[r] = blur_map(mask_f, color_f, bright_f, r)
        print(f"r={r}: max={maps[r].max():.3f}", flush=True)
    cv2.imwrite(os.path.join(OUT, "sig13.png"), (np.clip(maps[13], 0, 1) * 255).astype(np.uint8))

    summary = {}
    for slot in range(3):
        print(f"\n{'='*60}\nSLOT {slot} (verdade visual: {EXPECTED[slot]})\n{'='*60}", flush=True)
        sxn, syn = config.TRAY_SLOT_CENTERS[slot]
        sx, sy = sxn * sc, syn * (H / NATIVE_DEVICE_H)
        print(f"slot center=({sx:.1f},{sy:.1f}) pitch={pitch:.1f}", flush=True)

        print("A) signal BRUTO (mask_f 0/1 nos pontos da grade):", flush=True)
        eval_grid(mask_f, W, H, sx, sy, pitch, adj_thr=None, tag="bruto")
        for r in (5, 6, 7, 8, 9, 13):
            print(f"R={r} SEM adjacencia:", flush=True)
            det = eval_grid(maps[r], W, H, sx, sy, pitch, adj_thr=None, tag=f"r{r}")
            got = norm(det[0]) if det else []
            ok = got == EXPECTED[slot]
            summary.setdefault(r, {})[slot] = (len(got), got, ok)
            print(f"  -> r={r} slot{slot}: n={len(got)} {got} "
                  f"esperado={EXPECTED[slot]} {'OK' if ok else 'DIVERGE'}", flush=True)

    print(f"\n{'='*60}\nRESUMO (forma normalizada vs verdade visual)\n{'='*60}", flush=True)
    for r in (5, 6, 7, 8, 9, 13):
        line = f"r={r:2d}: " + " | ".join(
            f"S{s}:n={summary[r][s][0]}{'OK' if summary[r][s][2] else 'XX'}" for s in range(3))
        allok = all(summary[r][s][2] for s in range(3))
        print(line + ("   <<< TODOS CORRETOS" if allok else ""), flush=True)


if __name__ == "__main__":
    main()
