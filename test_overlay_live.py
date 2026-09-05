#!/usr/bin/env python3
"""
Overlay de diagnostico sobre captura salva (somente leitura + desenho).
Nao altera vision.py. Nenhuma jogada. Nenhum toque.
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from vision import (detect_board, read_pieces, _build_signal_map,
                    _snap_to_catalog, _normalize, _max_run,
                    _rescue_maps, _complete_winner_with_diamonds,
                    NATIVE_DEVICE_W, NATIVE_DEVICE_H)

CAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "logs", "diag_readonly", "live_20260904_185226.png")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "logs", "diag_readonly", "overlay_20260904_185226.png")


def mirror_winner(sig, W, H, sx, sy, sc):
    """Espelho exato da busca atual do vision.py (52-64, peso 1.0,
    strict-only, offsets ±8 com zero). Retorna vencedor + pontos."""
    cc = np.arange(7, dtype=np.float64) - 3.0
    rr = np.arange(7, dtype=np.float64) - 3.0
    best, best_score = None, float("-inf")
    for pitch in np.arange(52.0 * sc, 64.0 * sc, 1.0 * sc):
        half = int(pitch / 2) + 2
        base = np.arange(max(-half, -8.0), min(half, 8.0) + 1, 3, dtype=np.float64)
        dxa = np.union1d(base, np.array([0.0]))
        dya = dxa
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
                pts = np.zeros((r1 - r0, c1 - c0, 2), dtype=int)
                for a in range(r0, r1):
                    for q in range(c0, c1):
                        pts[a - r0, q - c0] = (int(xc[i, q]), int(yc[j, a]))
                best = (pitch, float(dxa[i]), float(dya[j]), bb.copy(),
                        bv.copy(), pts, (r0, r1, c0, c1), s)
    return best


def main():
    fr = cv2.imread(CAP_PATH)
    assert fr is not None, CAP_PATH
    H, W = fr.shape[:2]
    print(f"captura: {W}x{H}", flush=True)
    sc = W / NATIVE_DEVICE_W

    board = detect_board(fr)
    pieces = read_pieces(fr)
    sig = _build_signal_map(fr)
    rescue_maps = _rescue_maps(fr)

    vis = fr.copy()

    # 1. regiao do tabuleiro
    gx1, gy1 = config.GRID_TOP_LEFT
    gx2, gy2 = config.GRID_BOTTOM_RIGHT
    cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (255, 255, 255), 3)

    # 2. grade detectada + 3. pontos ocupado/vazio
    rows, cols = board.rows, board.cols
    px = (gx2 - gx1) / cols
    py = (gy2 - gy1) / rows
    for i in range(cols + 1):
        x = int(gx1 + i * px)
        cv2.line(vis, (x, gy1), (x, gy2), (180, 180, 180), 1)
    for i in range(rows + 1):
        y = int(gy1 + i * py)
        cv2.line(vis, (gx1, y), (gx2, y), (180, 180, 180), 1)
    for r in range(rows):
        for c in range(cols):
            cx = int(gx1 + (c + 0.5) * px)
            cy = int(gy1 + (r + 0.5) * py)
            if board.grid[r, c]:
                cv2.circle(vis, (cx, cy), 14, (0, 255, 0), -1)
            else:
                cv2.circle(vis, (cx, cy), 6, (100, 100, 100), -1)

    # 4/5/6. regioes das pecas + pontos amostrados + identificacao
    half = config.PIECE_SLOT_HALF
    for slot in range(3):
        sxn, syn = config.TRAY_SLOT_CENTERS[slot]
        sx, sy = sxn * sc, syn * (H / NATIVE_DEVICE_H)
        cv2.rectangle(vis, (int(sxn - half), int(syn - half)),
                      (int(sxn + half), int(syn + half)), (255, 160, 0), 2)
        w = mirror_winner(sig, W, H, sx, sy, sc)
        p = pieces[slot] if slot < len(pieces) else None
        if w is None:
            label = f"S{slot}: sem vencedor"
            cv2.putText(vis, label, (int(sxn - half) + 5, int(syn + half) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            print(f"slot{slot}: sem vencedor | detector={p.cells if p and p.cells else None}", flush=True)
            continue
        pitch, dx, dy, bb, bv, pts, (r0, r1, c0, c1), s = w
        # mesma completude do detector real (helper compartilhado)
        rescued_rel = []
        comp = _complete_winner_with_diamonds(
            sig, W, H, sx, sy, pitch, dx, dy, r0, c0, bb,
            slot, rescue_maps)
        if comp is not None:
            bb, r0, c0, rescued_rel = comp[0], comp[1], comp[2], comp[3]
            bh, bw = bb.shape
            # repoe pontos da grade e valores de sinal para o bbox expandido
            cc = np.arange(7, dtype=np.float64) - 3.0
            rr = np.arange(7, dtype=np.float64) - 3.0
            xi = np.round(sx + dx + cc * pitch).astype(np.int32)
            yi = np.round(sy + dy + rr * pitch).astype(np.int32)
            xic = np.clip(xi, 0, W - 1)
            yic = np.clip(yi, 0, H - 1)
            vals7 = sig[yic[:, None], xic[None, :]].astype(np.float64)
            pts = np.zeros((bh, bw, 2), dtype=int)
            bv = np.zeros((bh, bw), dtype=float)
            for r in range(bh):
                for c in range(bw):
                    pts[r, c] = (int(xic[c0 + c]), int(yic[r0 + r]))
                    bv[r, c] = float(vals7[r0 + r, c0 + c])
        bh, bw = bb.shape
        rescued_set = set(rescued_rel)
        for r in range(bh):
            for c in range(bw):
                px_, py_ = int(pts[r, c, 0]), int(pts[r, c, 1])
                if (r, c) in rescued_set:
                    cv2.circle(vis, (px_, py_), 18, (255, 0, 255), 3)
                    cv2.putText(vis, "DIAMANTE(rescue)", (px_ + 20, py_ - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
                elif bb[r, c]:
                    cv2.circle(vis, (px_, py_), 16, (0, 255, 0), 3)
                    cv2.putText(vis, f"{bv[r, c]:.2f}", (px_ + 18, py_),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.circle(vis, (px_, py_), 10, (0, 0, 255), 2)
        n = int(bb.sum())
        label = (f"S{slot}: {bw}x{bh} [{n}b] p={pitch:.0f} "
                 f"dx={dx:+.0f} dy={dy:+.0f} s={s:.2f}"
                 + (f" +rescue{sorted(rescued_rel)}" if rescued_rel else ""))
        cv2.putText(vis, label, (int(sxn - half) + 5, int(syn + half) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        det = "real" if (p and p.cells and sorted(p.cells) == sorted(
            [(r, c) for r in range(bh) for c in range(bw) if bb[r, c]])) else "?"
        print(f"slot{slot}: {bw}x{bh} [{n}b] pitch={pitch:.0f} dx={dx:+.0f} "
              f"dy={dy:+.0f} score={s:.4f} detector={p.cells if p and p.cells else None}",
              flush=True)

    cv2.imwrite(OUT_PATH, vis)
    print(f"OVERLAY: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
