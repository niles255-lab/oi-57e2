#!/usr/bin/env python3
"""
Compara limites de dx/dy no MESMO frame offline (diagnostico apenas).
Replica exata da busca atual do vision.py (52-64, peso pitch 1.0,
strict-only, r=6), variando so o dominio de offsets.
Nenhuma alteracao no codigo do jogo. Nenhuma jogada.
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from vision import (_build_signal_map, _snap_to_catalog, _normalize,
                    _max_run, NATIVE_DEVICE_W, NATIVE_DEVICE_H)

FRAME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "diag_diamond", "frame.png")
EXPECTED = {0: [(0, 0), (0, 1), (0, 2)],
            1: [(0, 0), (0, 1), (1, 1)],
            2: [(0, 0), (1, 0), (2, 0)]}


def search(sig, W, H, sx, sy, sc, lo, hi, extra_offsets=()):
    cc = np.arange(7, dtype=np.float64) - 3.0
    rr = np.arange(7, dtype=np.float64) - 3.0
    best, best_score = None, float("-inf")
    for pitch in np.arange(52.0 * sc, 64.0 * sc, 1.0 * sc):
        half = int(pitch / 2) + 2
        if extra_offsets:
            dxa = np.array(sorted(set(float(v) for v in extra_offsets)), dtype=np.float64)
            dya = dxa
        else:
            dxa = np.arange(max(-half, lo), min(half, hi) + 1, 3, dtype=np.float64)
            dya = np.arange(max(-half, lo), min(half, hi) + 1, 3, dtype=np.float64)
        if dxa.size == 0 or dya.size == 0:
            continue
        xf = sx + dxa[:, None] + cc[None, :] * pitch
        yf = sy + dya[:, None] + rr[None, :] * pitch
        xi = np.round(xf).astype(np.int32)
        yi = np.round(yf).astype(np.int32)
        xo = (xi < 0) | (xi >= W)
        yo = (yi < 0) | (yi >= H)
        xc = np.clip(xi, 0, W - 1)
        yc = np.clip(yi, 0, H - 1)
        va = sig[yc[np.newaxis, :, :, np.newaxis],
                 xc[:, np.newaxis, np.newaxis, :]].astype(np.float64)
        va[(xo[:, np.newaxis, np.newaxis, :] | yo[np.newaxis, :, :, np.newaxis])] = 0.0
        bits = va >= 0.60
        for idx in np.argwhere(bits.any(axis=(-1, -2))):
            i, j = int(idx[0]), int(idx[1])
            b = bits[i, j]
            v = va[i, j]
            rw = np.where(b.any(axis=1))[0]
            cw_ = np.where(b.any(axis=0))[0]
            if rw.size == 0 or cw_.size == 0:
                continue
            r0, r1 = int(rw[0]), int(rw[-1]) + 1
            c0, c1 = int(cw_[0]), int(cw_[-1]) + 1
            bb, bv = b[r0:r1, c0:c1], v[r0:r1, c0:c1]
            n = int(bb.sum())
            if n < 2 or n > 5:
                continue
            ms = float(bv[bb].mean())
            if ms < 0.55:
                continue
            bp = np.pad(bb.astype(np.int8), 1)
            nb = (bp[:-2, 1:-1] + bp[2:, 1:-1] + bp[1:-1, :-2] + bp[1:-1, 2:])
            sup = int(((nb > 0) & bb).sum())
            run = _max_run(bb)
            eib = (r1 - r0) * (c1 - c0) - n
            det = [(r, c) for r in range(r1 - r0) for c in range(c1 - c0) if bb[r, c]]
            sn = _snap_to_catalog(det)
            if len(sn) != n:
                continue
            rn_, sn_ = _normalize(det), _normalize(sn)
            jac = len(rn_ & sn_) / len(rn_ | sn_) if (rn_ | sn_) else 0
            if jac < 0.70:
                continue
            err = abs(pitch - 58.0 * sc) / 58.0
            s = (ms + 0.050 * sup + 0.040 * n + 0.100 * run - 0.080 * eib
                 - 1.0 * err - 0.15 * (5 - n) + 0.20 * jac)
            if s > best_score:
                best_score = s
                best = (pitch, float(dxa[i]), float(dya[j]), sorted(rn_),
                        sorted(sn_), n, ms, s)
    return best, best_score


def main():
    fr = cv2.imread(FRAME_PATH)
    assert fr is not None
    H, W = fr.shape[:2]
    print(f"frame: {W}x{H}", flush=True)
    sc = W / NATIVE_DEVICE_W
    sig = _build_signal_map(fr)
    experiments = [
        ("[-8,+8] passo 3 (sem o zero)", -8.0, 8.0, ()),
        ("[-8,-5,-2,0,1,4,7] (com o zero)", None, None, (-8.0, -5.0, -2.0, 0.0, 1.0, 4.0, 7.0)),
    ]
    for title, lo, hi, extra in experiments:
        print(f"\n{'#'*64}\nEXPERIMENTO {title}\n{'#'*64}", flush=True)
        for slot in range(3):
            sxn, syn = config.TRAY_SLOT_CENTERS[slot]
            sx, sy = sxn * sc, syn * (H / NATIVE_DEVICE_H)
            best, score = search(sig, W, H, sx, sy, sc, lo, hi, extra)
            # candidato correto ancorado: pitch 58, dx=0, dy=0
            ref, _ = search(sig, W, H, sx, sy, sc, 0.0, 0.0, (0.0,))
            if best is None:
                print(f"  slot{slot}: NENHUM vencedor | esperado={EXPECTED[slot]}", flush=True)
                continue
            pitch, dx, dy, pre, pos, n, ms, s = best
            ok = pre == EXPECTED[slot] and pos == EXPECTED[slot]
            print(f"  slot{slot}: pitch={pitch:.0f} dx={dx:+.0f} dy={dy:+.0f} "
                  f"pre={pre} n={n} score={s:.4f} pos={pos} "
                  f"esperado={EXPECTED[slot]} -> {'CORRETO' if ok else 'FANTASMA'}", flush=True)
            if ref is not None:
                _, _, _, rpre, rpos, rn, rms, rs = ref
                print(f"    correto ancorado (58,0,0): pre={rpre} n={rn} score={rs:.4f} "
                      f"pos={rpos} (delta={s - rs:+.4f})", flush=True)
            else:
                print("    correto ancorado (58,0,0): NEM GERADO", flush=True)


if __name__ == "__main__":
    main()
