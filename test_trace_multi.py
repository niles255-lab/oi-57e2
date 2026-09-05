#!/usr/bin/env python3
"""
Trace com múltiplos frames até achar um com signal map válido.
"""

import sys
import time
import subprocess
import ctypes
import os
import cv2
import numpy as np
import mss

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
import vision
from vision import (
    _build_signal_map,
    _block_mask,
    detect_piece,
    _snap_to_catalog,
    _normalize,
    _max_run,
    PIECE_CATALOG,
)

SCRCPY_WINDOW_TITLE = "scrcpy_trace_multi"
SCRCPY_SERIAL = config.ADB_DEVICE_ID
SCRCPY_MAX_SIZE = "1080"
MAX_FRAMES = 20
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "diag_trace")

os.makedirs(OUTPUT_DIR, exist_ok=True)


class Rect(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def find_scrcpy_window(title, timeout=15.0):
    user32 = ctypes.windll.user32
    start = time.time()
    while time.time() - start < timeout:
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            return hwnd
        time.sleep(0.1)
    return 0


def get_client_rect(hwnd):
    user32 = ctypes.windll.user32
    client_rect = Rect()
    if not user32.GetClientRect(hwnd, ctypes.byref(client_rect)):
        return 0, 0, 0, 0
    cw = client_rect.right - client_rect.left
    ch = client_rect.bottom - client_rect.top
    pt = Point(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return 0, 0, 0, 0
    return pt.x, pt.y, cw, ch


def wait_for_window_ready(hwnd, min_width=300, min_height=500, timeout=10.0):
    start = time.time()
    while time.time() - start < timeout:
        left, top, width, height = get_client_rect(hwnd)
        if width >= min_width and height >= min_height:
            return True
        time.sleep(0.2)
    return False


def start_scrcpy():
    scrcpy_path = config.SCRCPY_PATH
    cmd = [scrcpy_path, f"--window-title={SCRCPY_WINDOW_TITLE}",
           f"--max-size={SCRCPY_MAX_SIZE}", f"--serial={SCRCPY_SERIAL}"]
    print(f"[trace] Iniciando scrcpy")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def capture_frame(hwnd):
    left, top, width, height = get_client_rect(hwnd)
    if width == 0 or height == 0:
        return None
    monitor = {"left": left, "top": top, "width": width, "height": height}
    with mss.mss() as sct:
        img = sct.grab(monitor)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def main():
    print("=" * 70)
    print("TRACE MÚLTIPLOS FRAMES - Aguardando signal map válido")
    print("=" * 70)

    scrcpy_proc = start_scrcpy()
    try:
        hwnd = find_scrcpy_window(SCRCPY_WINDOW_TITLE, timeout=15.0)
        if not hwnd:
            print("[ERRO] Janela não encontrada")
            return

        if not wait_for_window_ready(hwnd):
            print("[ERRO] Janela não ficou pronta")
            return

        print("[trace] Janela pronta. Capturando frames até signal map ter sinal...")
        
        for frame_idx in range(MAX_FRAMES):
            frame = capture_frame(hwnd)
            if frame is None:
                time.sleep(0.1)
                continue
            
            sig = _build_signal_map(frame)
            sig_min, sig_max, sig_mean = sig.min(), sig.max(), sig.mean()
            
            print(f"Frame {frame_idx+1:2d}: shape={frame.shape}, sig range=[{sig_min:.4f}, {sig_max:.4f}], mean={sig_mean:.4f}")
            
            if sig_max > 0.5:  # Tem sinal decente
                print(f"\n>>> FRAME {frame_idx+1} TEM SINAL FORTE! Iniciando trace detalhado...\n")
                
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"good_frame_{frame_idx}.png"), frame)
                
                # Trace detalhado para este frame
                for slot_idx in range(3):
                    print(f"\n{'='*60}")
                    print(f"SLOT {slot_idx} - Frame {frame_idx+1}")
                    print(f"{'='*60}")
                    
                    # Replica a lógica com trace das rejeições
                    trace_slot(sig, slot_idx, frame.shape)
                break
            else:
                # Salva frame fraco para debug
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"weak_frame_{frame_idx}.png"), frame)
            
            time.sleep(0.1)
        else:
            print(f"\n[NÃO ENCONTRADO] Nenhum frame com signal forte em {MAX_FRAMES} tentativas")

    finally:
        scrcpy_proc.terminate()
        try:
            scrcpy_proc.wait(timeout=3)
        except:
            scrcpy_proc.kill()


def trace_slot(sig, slot_index, frame_shape):
    """Trace detalhado de um slot."""
    sx, sy = config.TRAY_SLOT_CENTERS[slot_index]
    H, W = sig.shape[:2]
    
    print(f"\nConfig: sx={sx}, sy={sy}, TRAY_CELL_PITCH={config.TRAY_CELL_PITCH}")
    print(f"Signal map: W={W}, H={H}")
    
    native_w, native_h = 1080, 2340
    scale_x = W / native_w
    scale_y = H / native_h
    pitch_eff = config.TRAY_CELL_PITCH * scale_x
    print(f"Pitch efetivo no frame: {pitch_eff:.1f} (nativo={config.TRAY_CELL_PITCH})")
    
    best_score = -1.0
    best_b = None
    rejection_log = []
    candidates_evaluated = 0

    cc_norm = np.arange(7, dtype=np.float64) - 3.0
    rr_norm = np.arange(7, dtype=np.float64) - 3.0

    for pitch in np.arange(50.0, 67.0, 1.0):
        half = int(pitch / 2) + 2
        dx_arr = np.arange(-half, half + 1, 3, dtype=np.float64)
        dy_arr = np.arange(-half, half + 1, 3, dtype=np.float64)

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

        oob = (x_oob[:, np.newaxis, np.newaxis, :] | y_oob[np.newaxis, :, :, np.newaxis])
        vals_all[oob] = 0.0

        strict = vals_all >= 0.60
        adj = np.zeros_like(strict)
        adj[:, :, :-1, :] |= strict[:, :, 1:, :]
        adj[:, :, 1:, :] |= strict[:, :, :-1, :]
        adj[:, :, :, :-1] |= strict[:, :, :, 1:]
        adj[:, :, :, 1:] |= strict[:, :, :, :-1]
        bits_all = strict | ((vals_all >= 0.40) & adj)

        for idx in np.argwhere(bits_all.any(axis=(-1, -2))):
            candidates_evaluated += 1
            i_dx, j_dy = int(idx[0]), int(idx[1])
            bits = bits_all[i_dx, j_dy]
            vals = vals_all[i_dx, j_dy]

            rows_w = np.where(bits.any(axis=1))[0]
            cols_w = np.where(bits.any(axis=0))[0]
            if rows_w.size == 0 or cols_w.size == 0:
                rejection_log.append(f"p={pitch:.1f} dx={i_dx} dy={j_dy}: REJECT rows/cols vazio")
                continue
            
            r0, r1 = int(rows_w[0]), int(rows_w[-1]) + 1
            c0, c1 = int(cols_w[0]), int(cols_w[-1]) + 1
            n_cells = int(bits[r0:r1, c0:c1].sum())
            if n_cells < 2 or n_cells > 5:
                continue

            # Recorte
            while c1 - c0 > 5:
                left_sum = int(bits[r0:r1, c0].sum())
                right_sum = int(bits[r0:r1, c1 - 1].sum())
                if left_sum == 0: c0 += 1
                elif right_sum == 0: c1 -= 1
                elif left_sum <= right_sum: c0 += 1
                else: c1 -= 1
            while r1 - r0 > 5:
                top_sum = int(bits[r0, c0:c1].sum())
                bot_sum = int(bits[r1 - 1, c0:c1].sum())
                if top_sum == 0: r0 += 1
                elif bot_sum == 0: r1 -= 1
                elif top_sum <= bot_sum: r0 += 1
                else: r1 -= 1

            b = bits[r0:r1, c0:c1]
            bv = vals[r0:r1, c0:c1]
            n = int(b.sum())
            if n < 2 or n > 5:
                continue

            mean_sig = float(bv[b].mean())
            if mean_sig < 0.55:
                rejection_log.append(f"p={pitch:.1f}: REJECT mean_sig={mean_sig:.3f} < 0.55")
                continue

            bp = np.pad(b.astype(np.int8), 1)
            nbrs = (bp[:-2, 1:-1] + bp[2:, 1:-1] + bp[1:-1, :-2] + bp[1:-1, 2:])
            support = int(((nbrs > 0) & b).sum())
            run = _max_run(b)
            bh2, bw2 = r1 - r0, c1 - c0
            empty_in_bbox = bh2 * bw2 - n
            pitch_err = abs(pitch - config.TRAY_CELL_PITCH) / config.TRAY_CELL_PITCH

            detected_cells = [(r, c) for r in range(bh2) for c in range(bw2) if b[r, c]]
            snapped = _snap_to_catalog(detected_cells)
            if len(snapped) != n:
                rejection_log.append(f"p={pitch:.1f}: REJECT snap n {n}->{len(snapped)}")
                continue
            
            raw_norm = _normalize(detected_cells)
            snap_norm = _normalize(snapped)
            jaccard = len(raw_norm & snap_norm) / len(raw_norm | snap_norm) if (raw_norm | snap_norm) else 0
            if jaccard < 0.70:
                rejection_log.append(f"p={pitch:.1f}: REJECT jaccard={jaccard:.3f} < 0.70")
                continue

            score = (
                mean_sig
                + 0.050 * support
                + 0.040 * n
                + 0.100 * run
                - 0.080 * empty_in_bbox
                - 0.100 * pitch_err
                - 0.15 * (5 - n)
                + 0.20 * jaccard
            )

            if score > best_score:
                best_score = score
                best_b = b.copy()

            if score > 0.75:
                rejection_log.append(f"  >>> CANDIDATO score={score:.3f} pitch={pitch:.1f} n={n} mean={mean_sig:.3f}")

    print(f"\n--- RESULTADO SLOT {slot_index} ---")
    print(f"Candidatos avaliados: {candidates_evaluated}")
    print(f"Melhor score: {best_score:.3f}")
    print(f"Threshold: 0.75")
    
    if best_score >= 0.75:
        print(f"[OK] Score >= 0.75 - ACEITA")
    else:
        print(f"[REJECT] Score < 0.75")
    
    # Mostra rejeições importantes
    high_score_logs = [l for l in rejection_log if ">>>" in l]
    if high_score_logs:
        print(f"\nCandidatos com score > 0.75:")
        for log in high_score_logs[:10]:
            print(log)
    
    # Verifica etapa final do código real
    if best_score >= 0.75 and best_b is not None:
        print("\n--- SIMULANDO CÓDIGO REAL PÓS-LOOP ---")
        bh, bw = best_b.shape
        mask5 = np.zeros((5, 5), dtype=np.uint8)
        r_off = (5 - bh) // 2
        c_off = (5 - bw) // 2
        mask5[r_off:r_off + bh, c_off:c_off + bw] = best_b.astype(np.uint8)

        piece = vision.Piece.from_mask(piece_id=slot_index, mask=mask5)
        piece.tray_cx = float(sx)
        piece.tray_cy = float(sy)

        snapped_cells = _snap_to_catalog(piece.cells)
        if snapped_cells != piece.cells:
            piece2 = vision.Piece(
                id=slot_index, cells=snapped_cells,
                width=max(c for _, c in snapped_cells) + 1 if snapped_cells else 0,
                height=max(r for r, _ in snapped_cells) + 1 if snapped_cells else 0,
            )
            piece2.tray_cx = float(sx)
            piece2.tray_cy = float(sy)
            piece = piece2

        if len(piece.cells) < 2 or len(piece.cells) > 5:
            print(f"REJECT FINAL: piece.cells={len(piece.cells)} fora [2,5]")
        else:
            print(f"PEÇA FINAL ACEITA: {piece.width}x{piece.height}, {len(piece.cells)} blocos")
            print(f"  cells: {piece.cells}")


if __name__ == "__main__":
    main()
