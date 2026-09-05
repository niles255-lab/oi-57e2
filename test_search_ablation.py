#!/usr/bin/env python3
"""
Ablacao da busca de candidatos (diagnostico apenas).
TESTE A: pitch=58, dx=0, dy=0
TESTE B: pitch 52-64, dx=0, dy=0
TESTE C: pitch=58, dx/dy faixa atual
Mesmo frame offline. Nenhuma alteracao no codigo do jogo. Nenhuma jogada.
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

EXPECTED = {
    0: [(0, 0), (0, 1), (0, 2)],
    1: [(0, 0), (0, 1), (1, 1)],
    2: [(0, 0), (1, 0), (2, 0)],
}

PITCH_NATIVE = config.TRAY_CELL_PITCH  # 58
PITCH_W = 1.0  # igual ao vision.py atual


def run_search(sig, W, H, sx, sy, sc, pitches, dx_vals, dy_vals, tag):
    """Replica exata da busca do detect_piece (gates + score atuais)."""
    cc_norm = np.arange(7, dtype=np.float64) - 3.0
    rr_norm = np.arange(7, dtype=np.float64) - 3.0
    best = None
    best_score = float("-inf")
    truth_best = float("-inf")
    truth_seen = False
    n_eval = 0
    for pitch in pitches:
        half = int(pitch / 2) + 2
        dx_arr = np.array([v for v in dx_vals if -half <= v <= half], dtype=np.float64)
        dy_arr = np.array([v for v in dy_vals if -half <= v <= half], dtype=np.float64)
        if dx_arr.size == 0 or dy_arr.size == 0:
            continue
        all_xi_f = sx + dx_arr[:, None] + cc_norm[None, :] * pitch
        all_yi_f = sy + dy_arr[:, None] + rr_norm[None, :] * pitch
        all_xi = np.round(all_xi_f).astype(np.int32)
        all_yi = np.round(all_yi_f).astype(np.int32)
        x_oob = (all_xi < 0) | (all_xi >= W)
        y_oob = (all_yi < 0) | (all_yi >= H)
        xi_c = np.clip(all_xi, 0, W - 1)
        yi_c = np.clip(all_yi, 0, H - 1)
        vals_all = sig[
            yi_c[np.newaxis, :, :, np.newaxis],
            xi_c[:, np.newaxis, np.newaxis, :],
        ].astype(np.float64)
        oob = (x_oob[:, np.newaxis, np.newaxis, :]
               | y_oob[np.newaxis, :, :, np.newaxis])
        vals_all[oob] = 0.0
        strict = vals_all >= 0.60
        bits_all = strict  # adjacencia OFF (estado atual do codigo)
        for idx in np.argwhere(bits_all.any(axis=(-1, -2))):
            i_dx, j_dy = int(idx[0]), int(idx[1])
            bits = bits_all[i_dx, j_dy]
            vals = vals_all[i_dx, j_dy]
            rows_w = np.where(bits.any(axis=1))[0]
            cols_w = np.where(bits.any(axis=0))[0]
            if rows_w.size == 0 or cols_w.size == 0:
                continue
            r0, r1 = int(rows_w[0]), int(rows_w[-1]) + 1
            c0, c1 = int(cols_w[0]), int(cols_w[-1]) + 1
            b = bits[r0:r1, c0:c1]
            bv = vals[r0:r1, c0:c1]
            n = int(b.sum())
            if n < 2 or n > 5:
                continue
            n_eval += 1
            ms = float(bv[b].mean())
            if ms < 0.55:
                continue
            bp = np.pad(b.astype(np.int8), 1)
            nbrs = (bp[:-2, 1:-1] + bp[2:, 1:-1] + bp[1:-1, :-2] + bp[1:-1, 2:])
            support = int(((nbrs > 0) & b).sum())
            run = _max_run(b)
            bh2, bw2 = r1 - r0, c1 - c0
            eib = bh2 * bw2 - n
            det = [(r, c) for r in range(bh2) for c in range(bw2) if b[r, c]]
            snapped = _snap_to_catalog(det)
            if len(snapped) != n:
                continue
            rn_ = _normalize(det)
            sn_ = _normalize(snapped)
            jac = len(rn_ & sn_) / len(rn_ | sn_) if (rn_ | sn_) else 0
            if jac < 0.70:
                continue
            pitch_err = abs(pitch - PITCH_NATIVE * sc) / PITCH_NATIVE
            score = (ms + 0.050 * support + 0.040 * n + 0.100 * run
                     - 0.080 * eib - PITCH_W * pitch_err
                     - 0.15 * (5 - n) + 0.20 * jac)
            if sorted(rn_) == EXPECTED[slot]:
                truth_seen = True
                truth_best = max(truth_best, score)
            if score > best_score:
                best_score = score
                best = (pitch, float(dx_arr[i_dx]), float(dy_arr[j_dy]),
                        sorted(rn_), n, ms, score)
    return best, best_score, truth_seen, truth_best, n_eval


slot = 0  # placeholder global p/ closure simples


def main():
    fr = cv2.imread(FRAME_PATH)
    assert fr is not None, FRAME_PATH
    H, W = fr.shape[:2]
    print(f"frame: {W}x{H}", flush=True)
    sc = W / NATIVE_DEVICE_W
    sig = _build_signal_map(fr)

    global slot
    for slot in range(3):
        print(f"\n{'='*64}\nSLOT {slot} (verdade: {EXPECTED[slot]})\n{'='*64}", flush=True)
        sxn, syn = config.TRAY_SLOT_CENTERS[slot]
        sx, sy = sxn * sc, syn * (H / NATIVE_DEVICE_H)
        p58 = 58.0 * sc
        half58 = int(p58 / 2) + 2
        full = np.arange(-half58, half58 + 1, 3, dtype=np.float64)
        var = np.arange(52.0 * sc, 64.0 * sc, 1.0 * sc)

        print("TESTE A — pitch=58, dx=0, dy=0:", flush=True)
        b, s, ts, tbs, ne = run_search(sig, W, H, sx, sy, sc, [p58], [0.0], [0.0], "A")
        rep("A", b, s, ts, tbs, ne, slot)

        print("TESTE B — pitch 52-64, dx=0, dy=0:", flush=True)
        b, s, ts, tbs, ne = run_search(sig, W, H, sx, sy, sc, list(var), [0.0], [0.0], "B")
        rep("B", b, s, ts, tbs, ne, slot)

        print("TESTE C — pitch=58, dx/dy faixa atual:", flush=True)
        b, s, ts, tbs, ne = run_search(sig, W, H, sx, sy, sc, [p58], list(full), list(full), "C")
        rep("C", b, s, ts, tbs, ne, slot)


def rep(tag, best, score, truth_seen, truth_best, n_eval, slot):
    if best is None:
        print(f"  [{tag}] nenhum vencedor | verdade gerada={truth_seen} "
              f"(score verdade={truth_best:.4f}) | avaliados={n_eval}", flush=True)
        return
    pitch, dx, dy, shape, n, ms, sc_ = best
    ok = shape == EXPECTED[slot]
    print(f"  [{tag}] vencedor n={n} {shape} pitch={pitch:.1f} dx={dx:.0f} dy={dy:.0f} "
          f"score={score:.4f} | verdade gerada={truth_seen} "
          f"(score verdade={truth_best:.4f}) | avaliados={n_eval} -> "
          f"{'CORRETO' if ok else 'FANTASMA'}", flush=True)


if __name__ == "__main__":
    main()
