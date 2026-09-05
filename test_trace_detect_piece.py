#!/usr/bin/env python3
"""
Diagnóstico rastreando TODAS as condições de rejeição dentro do detect_piece() REAL.
Não altera vision.py — apenas instrumenta a execução.
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

SCRCPY_WINDOW_TITLE = "scrcpy_trace_diagnose"
SCRCPY_SERIAL = config.ADB_DEVICE_ID
SCRCPY_MAX_SIZE = "1080"
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
            print(f"[trace] Janela pronta: client area {width}x{height}")
            return True
        time.sleep(0.2)
    print(f"[trace] Timeout: janela não atingiu tamanho mínimo")
    return False


def start_scrcpy():
    scrcpy_path = config.SCRCPY_PATH
    cmd = [scrcpy_path, f"--window-title={SCRCPY_WINDOW_TITLE}",
           f"--max-size={SCRCPY_MAX_SIZE}", f"--serial={SCRCPY_SERIAL}"]
    print(f"[trace] Iniciando scrcpy: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def capture_one_frame(hwnd):
    if not wait_for_window_ready(hwnd):
        return None
    left, top, width, height = get_client_rect(hwnd)
    if width == 0 or height == 0:
        return None
    monitor = {"left": left, "top": top, "width": width, "height": height}
    with mss.mss() as sct:
        img = sct.grab(monitor)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


# ============================================================
# INSTRUMENTAÇÃO DO detect_piece() REAL
# ============================================================

original_detect_piece = vision.detect_piece

def traced_detect_piece(sig, slot_index):
    print(f"\n{'='*70}")
    print(f"TRACE detect_piece(sig, slot={slot_index})")
    print(f"{'='*70}")
    print(f"Frame shape (via sig): {sig.shape}")
    
    sx, sy = config.TRAY_SLOT_CENTERS[slot_index]
    H, W = sig.shape[:2]
    
    print(f"Config: sx={sx}, sy={sy}, TRAY_CELL_PITCH={config.TRAY_CELL_PITCH}")
    print(f"Signal map: W={W}, H={H}")
    
    # Calcula pitch efetivo no frame
    native_w, native_h = 1080, 2340
    scale_x = W / native_w
    scale_y = H / native_h
    pitch_eff = config.TRAY_CELL_PITCH * scale_x
    print(f"Escala: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")
    print(f"Pitch efetivo no frame: {pitch_eff:.1f}")
    
    # Chama a função original mas com tracing interno
    # Vamos replicar a lógica COM TRACE de cada condição
    
    best_score = -1.0
    best_b = None
    best_pitch = config.TRAY_CELL_PITCH
    rejection_log = []
    candidates_evaluated = 0
    candidates_passed_basic = 0

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
                rejection_log.append(f"  pitch={pitch:.1f} dx={i_dx} dy={j_dy}: REJECT - rows_w/cols_w vazio")
                continue
            
            r0, r1 = int(rows_w[0]), int(rows_w[-1]) + 1
            c0, c1 = int(cols_w[0]), int(cols_w[-1]) + 1

            n_cells = int(bits[r0:r1, c0:c1].sum())
            if n_cells < 2 or n_cells > 5:
                rejection_log.append(f"  pitch={pitch:.1f} dx={i_dx} dy={j_dy}: REJECT - n_cells={n_cells} (min=2 max=5)")
                continue

            # Recorte agressivo (pode remover células!)
            while c1 - c0 > 5:
                left_sum = int(bits[r0:r1, c0].sum())
                right_sum = int(bits[r0:r1, c1 - 1].sum())
                if left_sum == 0:
                    c0 += 1
                elif right_sum == 0:
                    c1 -= 1
                elif left_sum <= right_sum:
                    c0 += 1
                else:
                    c1 -= 1
            while r1 - r0 > 5:
                top_sum = int(bits[r0, c0:c1].sum())
                bot_sum = int(bits[r1 - 1, c0:c1].sum())
                if top_sum == 0:
                    r0 += 1
                elif bot_sum == 0:
                    r1 -= 1
                elif top_sum <= bot_sum:
                    r0 += 1
                else:
                    r1 -= 1

            b = bits[r0:r1, c0:c1]
            bv = vals[r0:r1, c0:c1]
            n = int(b.sum())
            if n < 2 or n > 5:
                rejection_log.append(f"  pitch={pitch:.1f} dx={i_dx} dy={j_dy}: REJECT pós-recorte n={n}")
                continue

            candidates_passed_basic += 1
            
            mean_sig = float(bv[b].mean())
            if mean_sig < 0.55:
                rejection_log.append(f"  pitch={pitch:.1f} dx={i_dx} dy={j_dy}: REJECT mean_sig={mean_sig:.3f} < 0.55")
                continue

            bp = np.pad(b.astype(np.int8), 1)
            nbrs = (bp[:-2, 1:-1] + bp[2:, 1:-1] + bp[1:-1, :-2] + bp[1:-1, 2:])
            support = int(((nbrs > 0) & b).sum())

            run = _max_run(b)
            bh2, bw2 = r1 - r0, c1 - c0
            empty_in_bbox = bh2 * bw2 - n

            pitch_err = abs(pitch - config.TRAY_CELL_PITCH) / config.TRAY_CELL_PITCH

            # Snap to catalog + Jaccard
            detected_cells = [(r, c) for r in range(bh2) for c in range(bw2) if b[r, c]]
            snapped = _snap_to_catalog(detected_cells)
            if len(snapped) != n:
                rejection_log.append(f"  pitch={pitch:.1f} dx={i_dx} dy={j_dy}: REJECT snap mudou n ({n}->{len(snapped)})")
                continue
            
            raw_norm = _normalize(detected_cells)
            snap_norm = _normalize(snapped)
            jaccard = len(raw_norm & snap_norm) / len(raw_norm | snap_norm) if (raw_norm | snap_norm) else 0
            if jaccard < 0.70:
                rejection_log.append(f"  pitch={pitch:.1f} dx={i_dx} dy={j_dy}: REJECT jaccard={jaccard:.3f} < 0.70")
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
                best_pitch = pitch

            # Log detalhado para candidatos com score alto
            if score > 0.75:
                rejection_log.append(f"  >>> CANDIDATO ALTO pitch={pitch:.1f} n={n} mean_sig={mean_sig:.3f} "
                                   f"support={support} run={run} empty={empty_in_bbox} "
                                   f"jaccard={jaccard:.3f} score={score:.3f} [PASSOU SCORE]")

    print(f"\n--- RESUMO DO TRACE ---")
    print(f"Candidatos avaliados (bits_all.any): {candidates_evaluated}")
    print(f"Candidatos passaram básico (n_cells 2-5): {candidates_passed_basic}")
    print(f"Melhor score final: {best_score:.3f}")
    print(f"Threshold mínimo: 0.75")
    
    if best_score >= 0.75:
        print(f"[OK] Score >= 0.75 - deveria aceitar!")
    else:
        print(f"[REJECT] Score < 0.75 - rejeitado por score final")

    print(f"\n--- LOG DE REJEIÇÕES (primeiras 30) ---")
    for log in rejection_log[:30]:
        print(log)
    if len(rejection_log) > 30:
        print(f"  ... mais {len(rejection_log) - 30} rejeições")

    # Agora verifica o que o código REAL faz depois
    print(f"\n--- VERIFICAÇÃO PÓS-LOOP (código real) ---")
    if best_b is None or best_b.sum() == 0:
        print(f"REJECT: best_b is None or sum=0")
        return None

    # Score mínimo final (linha 408-411 no vision.py atual)
    if best_score < 0.75:
        print(f"REJECT: best_score={best_score:.3f} < 0.75 (threshold final)")
        return None

    print(f"[OK] Passou threshold final 0.75")

    # Resto do código real
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
            id=slot_index,
            cells=snapped_cells,
            width=max(c for _, c in snapped_cells) + 1 if snapped_cells else 0,
            height=max(r for r, _ in snapped_cells) + 1 if snapped_cells else 0,
        )
        piece2.tray_cx = float(sx)
        piece2.tray_cy = float(sy)
        piece = piece2

    # Validação final (linhas 434-439 no vision.py atual)
    if len(piece.cells) < 2 or len(piece.cells) > 5:
        print(f"REJECT final: piece.cells={len(piece.cells)} fora de [2,5]")
        return None

    print(f"[OK] Peça final aceita: {piece.width}x{piece.height}, {len(piece.cells)} blocos")
    return piece


# Monkey patch temporário
vision.detect_piece = traced_detect_piece


def main():
    print("=" * 70)
    print("TRACE COMPLETO DO detect_piece() REAL")
    print("=" * 70)

    scrcpy_proc = start_scrcpy()
    try:
        hwnd = find_scrcpy_window(SCRCPY_WINDOW_TITLE, timeout=15.0)
        if not hwnd:
            print("[ERRO] Janela não encontrada")
            return

        print("[trace] Capturando 1 frame para trace...")
        frame = capture_one_frame(hwnd)
        if frame is None:
            print("[ERRO] Frame não capturado")
            return

        print(f"[trace] Frame capturado: {frame.shape}")
        cv2.imwrite(os.path.join(OUTPUT_DIR, "trace_full.png"), frame)

        sig = _build_signal_map(frame)
        print(f"[trace] Signal map: {sig.shape}, range=[{sig.min():.4f}, {sig.max():.4f}]")

        for slot_idx in range(3):
            print(f"\n\n{'#'*70}")
            print(f"### SLOT {slot_idx}")
            print(f"{'#'*70}")
            
            piece = traced_detect_piece(sig, slot_idx)
            if piece:
                print(f"\n>>> RESULTADO FINAL SLOT {slot_idx}: PEÇA DETECTADA")
                print(f"    {piece.width}x{piece.height}, {len(piece.cells)} blocos")
                print(f"    cells: {piece.cells}")
            else:
                print(f"\n>>> RESULTADO FINAL SLOT {slot_idx}: NONE")

    finally:
        scrcpy_proc.terminate()
        try:
            scrcpy_proc.wait(timeout=3)
        except:
            scrcpy_proc.kill()

    print(f"\n{'='*70}")
    print("TRACE CONCLUÍDO")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
