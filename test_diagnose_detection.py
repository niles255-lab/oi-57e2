#!/usr/bin/env python3
"""
Diagnóstico isolado: por que read_pieces() retorna None para os 3 slots?
Pipeline: scrcpy -> mss -> NumPy/OpenCV -> vision.py (diagnóstico detalhado)
NÃO ALTERA NENHUM ARQUIVO DO PROJETO PRINCIPAL.
"""

import sys
import time
import subprocess
import threading
import queue
import ctypes
import os
import cv2
import numpy as np
import mss

# Adiciona pasta do projeto ao path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Importa visão e config
import config
from vision import (
    _build_signal_map,
    _block_mask,
    detect_piece,
    _snap_to_catalog,
    _normalize,
    _max_run,
    PIECE_CATALOG,
)
from model import Piece

# Configurações do teste
SCRCPY_WINDOW_TITLE = "scrcpy_diagnose_detection"
SCRCPY_SERIAL = config.ADB_DEVICE_ID
SCRCPY_MAX_SIZE = "1080"
NUM_FRAMES_TO_ANALYZE = 3
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "diag_output")

os.makedirs(OUTPUT_DIR, exist_ok=True)


class Rect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def find_scrcpy_window(title: str, timeout: float = 15.0) -> int:
    user32 = ctypes.windll.user32
    start = time.time()
    while time.time() - start < timeout:
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            return hwnd
        time.sleep(0.1)
    return 0


def get_client_rect(hwnd: int) -> tuple:
    """Retorna (left, top, width, height) da área cliente (vídeo) em coordenadas de tela."""
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


def wait_for_window_ready(hwnd: int, min_width: int = 300, min_height: int = 500, timeout: float = 10.0) -> bool:
    """Aguarda a área cliente da janela atingir tamanho mínimo razoável."""
    start = time.time()
    while time.time() - start < timeout:
        left, top, width, height = get_client_rect(hwnd)
        if width >= min_width and height >= min_height:
            print(f"[diag] Janela pronta: client area {width}x{height}")
            return True
        time.sleep(0.2)
    print(f"[diag] Timeout: janela não atingiu tamanho mínimo ({min_width}x{min_height})")
    return False


def start_scrcpy():
    scrcpy_path = config.SCRCPY_PATH
    cmd = [
        scrcpy_path,
        f"--window-title={SCRCPY_WINDOW_TITLE}",
        f"--max-size={SCRCPY_MAX_SIZE}",
        f"--serial={SCRCPY_SERIAL}",
    ]
    print(f"[diag] Iniciando scrcpy: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def capture_frames(hwnd, num_frames):
    if not wait_for_window_ready(hwnd):
        print("[diag] Janela não ficou pronta a tempo")
        return []

    left, top, width, height = get_client_rect(hwnd)
    if width == 0 or height == 0:
        print("[diag] Área cliente inválida")
        return []

    monitor = {"left": left, "top": top, "width": width, "height": height}
    print(f"[diag] Capturando {num_frames} frames de {width}x{height} (client area)...")

    frames = []
    with mss.mss() as sct:
        for i in range(num_frames):
            img = sct.grab(monitor)
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frames.append(frame)
            time.sleep(0.05)  # pequeno intervalo
    return frames


def analyze_slot(frame, slot_index, frame_idx):
    """Analisa um slot específico em detalhes."""
    print(f"\n{'='*70}")
    print(f"FRAME {frame_idx} | SLOT {slot_index}")
    print(f"{'='*70}")

    # 1. Coordenadas originais do config
    sx_orig, sy_orig = config.TRAY_SLOT_CENTERS[slot_index]
    print(f"\n1. COORDENADAS ORIGINAIS (config.py):")
    print(f"   TRAY_SLOT_CENTERS[{slot_index}] = ({sx_orig}, {sy_orig})")
    print(f"   TRAY_CELL_PITCH = {config.TRAY_CELL_PITCH}")
    print(f"   PIECE_SLOT_HALF = {config.PIECE_SLOT_HALF}")

    # 2. Calcula escala baseada no frame capturado REAL
    # O vision.py espera coordenadas na resolução nativa do device
    # Precisamos mapear as coordenadas do config para o frame capturado
    frame_h, frame_w = frame.shape[:2]
    
    # Assume resolução nativa do Redmi Note 8: 1080x2340
    # mas calcula escala real baseada no frame capturado
    native_w, native_h = 1080, 2340
    scale_x = frame_w / native_w
    scale_y = frame_h / native_h
    print(f"\n2. ESCALA CALCULADA:")
    print(f"   Frame capturado: {frame_w}x{frame_h}")
    print(f"   Nativa assumida: {native_w}x{native_h}")
    print(f"   scale_x = {scale_x:.4f}, scale_y = {scale_y:.4f}")
    print(f"   Aspecto frame: {frame_w/frame_h:.3f}, Aspecto nativo: {native_w/native_h:.3f}")

    sx_eff = int(sx_orig * scale_x)
    sy_eff = int(sy_orig * scale_y)
    half_eff = int(config.PIECE_SLOT_HALF * scale_x)
    pitch_eff = config.TRAY_CELL_PITCH * scale_x

    print(f"\n3. COORDENADAS EFETIVAS NO FRAME {frame_w}x{frame_h}:")
    print(f"   Centro do slot: ({sx_eff}, {sy_eff})")
    print(f"   Half size: {half_eff}")
    print(f"   Pitch efetivo: {pitch_eff:.1f}")
    print(f"   Região do slot: x=[{sx_eff-half_eff}:{sx_eff+half_eff}], y=[{sy_eff-half_eff}:{sy_eff+half_eff}]")

    # Verifica limites
    if sx_eff - half_eff < 0 or sy_eff - half_eff < 0 or sx_eff + half_eff >= frame_w or sy_eff + half_eff >= frame_h:
        print(f"   [WARN] SLOT FORA DOS LIMITES DO FRAME!")
        # Não retorna - continua para ver o que acontece

    # 4. Recorte visual do slot
    slot_crop = frame[sy_eff-half_eff:sy_eff+half_eff, sx_eff-half_eff:sx_eff+half_eff].copy()
    print(f"\n4. RECORTE DO SLOT: {slot_crop.shape}")

    # Salva recorte
    crop_path = os.path.join(OUTPUT_DIR, f"frame{frame_idx}_slot{slot_index}_crop.png")
    cv2.imwrite(crop_path, slot_crop)
    print(f"   Salvo em: {crop_path}")

    # 5. Signal map
    print(f"\n5. SIGNAL MAP (_build_signal_map):")
    sig = _build_signal_map(frame)
    print(f"   Shape do signal map: {sig.shape}")
    print(f"   Range: [{sig.min():.4f}, {sig.max():.4f}], média: {sig.mean():.4f}")

    # Estatísticas na região do slot
    y1, y2 = max(0, sy_eff-half_eff), min(frame_h, sy_eff+half_eff)
    x1, x2 = max(0, sx_eff-half_eff), min(frame_w, sx_eff+half_eff)
    slot_sig = sig[y1:y2, x1:x2]
    print(f"   Região analisada no signal map: y=[{y1}:{y2}], x=[{x1}:{x2}] -> shape {slot_sig.shape}")
    print(f"   Stats na região do slot:")
    print(f"     min: {slot_sig.min():.4f}")
    print(f"     max: {slot_sig.max():.4f}")
    print(f"     média: {slot_sig.mean():.4f}")
    print(f"     pixels > 0.1: {np.sum(slot_sig > 0.1)} / {slot_sig.size} ({100*np.sum(slot_sig > 0.1)/slot_sig.size:.1f}%)")
    print(f"     pixels > 0.3: {np.sum(slot_sig > 0.3)} / {slot_sig.size} ({100*np.sum(slot_sig > 0.3)/slot_sig.size:.1f}%)")
    print(f"     pixels > 0.5: {np.sum(slot_sig > 0.5)} / {slot_sig.size} ({100*np.sum(slot_sig > 0.5)/slot_sig.size:.1f}%)")

    # 6. HSV e block mask na região
    print(f"\n6. HSV E BLOCK MASK NA REGIÃO DO SLOT:")
    slot_hsv = cv2.cvtColor(slot_crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(slot_hsv)
    print(f"   H: min={h.min()}, max={h.max()}, mean={h.mean():.1f}")
    print(f"   S: min={s.min()}, max={s.max()}, mean={s.mean():.1f}")
    print(f"   V: min={v.min()}, max={v.max()}, mean={v.mean():.1f}")

    mask = _block_mask(slot_hsv)
    mask_ratio = np.sum(mask > 0) / mask.size
    print(f"   Block mask ratio: {mask_ratio:.3f} ({np.sum(mask > 0)}/{mask.size})")

    # Salva mask
    mask_path = os.path.join(OUTPUT_DIR, f"frame{frame_idx}_slot{slot_index}_mask.png")
    cv2.imwrite(mask_path, mask)

    # 7. Executa detect_piece com debug
    print(f"\n7. EXECUTANDO detect_piece(sig, {slot_index}):")
    try:
        piece = detect_piece(sig, slot_index)
        if piece:
            print(f"   [OK] PEÇA DETECTADA: {piece.width}x{piece.height}, blocos={len(piece.cells)}")
            print(f"      cells: {piece.cells}")
            print(f"      pitch: {getattr(piece, 'tray_cx', 'N/A')}")
        else:
            print(f"   [FAIL] RETORNOU None")

        # Tenta chamar detect_piece com mais detalhes - vamos replicar a lógica interna
        print(f"\n8. ANÁLISE DETALHADA DA DETECÇÃO INTERNA:")
        analyze_detect_piece_internals(sig, slot_index, sx_eff, sy_eff, pitch_eff, frame_w, frame_h)

    except Exception as e:
        print(f"   ERRO em detect_piece: {e}")
        import traceback
        traceback.print_exc()


def analyze_detect_piece_internals(sig, slot_index, sx, sy, pitch, W, H):
    """Replica a lógica interna de detect_piece para diagnóstico."""
    print(f"   Parâmetros: sx={sx}, sy={sy}, pitch={pitch:.1f}, W={W}, H={H}")

    cc_norm = np.arange(7, dtype=np.float64) - 3.0
    rr_norm = np.arange(7, dtype=np.float64) - 3.0

    best_score = -1.0
    best_b = None
    best_pitch = pitch
    candidates_found = 0

    for p in np.arange(50.0, 67.0, 1.0):
        half = int(p / 2) + 2
        dx_arr = np.arange(-half, half + 1, 3, dtype=np.float64)
        dy_arr = np.arange(-half, half + 1, 3, dtype=np.float64)

        all_xi_f = sx + dx_arr[:, None] + cc_norm[None, :] * p
        all_yi_f = sy + dy_arr[:, None] + rr_norm[None, :] * p
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

        # Thresholds do código atual
        strict = vals_all >= 0.60
        adj = np.zeros_like(strict)
        adj[:, :, :-1, :] |= strict[:, :, 1:, :]
        adj[:, :, 1:, :] |= strict[:, :, :-1, :]
        adj[:, :, :, :-1] |= strict[:, :, :, 1:]
        adj[:, :, :, 1:] |= strict[:, :, :, :-1]
        bits_all = strict | ((vals_all >= 0.40) & adj)

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

            n_cells = int(bits[r0:r1, c0:c1].sum())
            if n_cells < 2 or n_cells > 5:
                continue

            b = bits[r0:r1, c0:c1]
            bv = vals[r0:r1, c0:c1]
            n = int(b.sum())
            if n < 2 or n > 5:
                continue

            mean_sig = float(bv[b].mean())
            if mean_sig < 0.55:
                continue

            bp = np.pad(b.astype(np.int8), 1)
            nbrs = (bp[:-2, 1:-1] + bp[2:, 1:-1] + bp[1:-1, :-2] + bp[1:-1, 2:])
            support = int(((nbrs > 0) & b).sum())

            run = _max_run(b)
            bh2, bw2 = r1 - r0, c1 - c0
            empty_in_bbox = bh2 * bw2 - n

            pitch_err = abs(p - config.TRAY_CELL_PITCH) / config.TRAY_CELL_PITCH

            # Jaccard com catálogo
            detected_cells = [(r, c) for r in range(bh2) for c in range(bw2) if b[r, c]]
            snapped = _snap_to_catalog(detected_cells)
            if len(snapped) != n:
                continue
            raw_norm = _normalize(detected_cells)
            snap_norm = _normalize(snapped)
            jaccard = len(raw_norm & snap_norm) / len(raw_norm | snap_norm) if (raw_norm | snap_norm) else 0
            if jaccard < 0.70:
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
                best_pitch = p

            candidates_found += 1
            if candidates_found <= 5:  # Mostra só os primeiros 5
                print(f"     Candidato {candidates_found}: pitch={p:.1f}, n={n}, mean_sig={mean_sig:.3f}, "
                      f"support={support}, run={run}, empty={empty_in_bbox}, jaccard={jaccard:.3f}, "
                      f"score={score:.3f}")

    print(f"\n   RESUMO DA DETECÇÃO:")
    print(f"     Total de candidatos avaliados: {candidates_found}")
    print(f"     Melhor score: {best_score:.3f}")
    print(f"     Threshold mínimo (código): 0.75")
    if best_b is not None:
        print(f"     [OK] Aceitaria (score >= 0.75)")
    else:
        print(f"     [REJECT] REJEITADO (nenhum candidato passou score >= 0.75)")

    # 9. Thresholds usados
    print(f"\n9. THRESHOLDS/VALORES MÍNIMOS DO CÓDIGO ATUAL:")
    print(f"     strict threshold: 0.60")
    print(f"     adj threshold: 0.40")
    print(f"     min cells: 2")
    print(f"     max cells: 5")
    print(f"     mean_sig mínimo: 0.55")
    print(f"     Jaccard mínimo: 0.70")
    print(f"     Score final mínimo: 0.75")
    print(f"     Pitch range testado: 50.0 - 67.0")


def main():
    print("=" * 70)
    print("DIAGNÓSTICO DETALHADO: vision.py.read_pieces() -> None")
    print("=" * 70)

    # Inicia scrcpy
    scrcpy_proc = start_scrcpy()

    try:
        # Aguarda janela
        print("[diag] Aguardando janela...")
        hwnd = find_scrcpy_window(SCRCPY_WINDOW_TITLE, timeout=15.0)
        if not hwnd:
            print("[ERRO] Janela não encontrada")
            return

        left, top, width, height = get_client_rect(hwnd)
        print(f"[diag] Janela (client area): {width}x{height} em ({left},{top})")

        # Captura frames
        frames = capture_frames(hwnd, NUM_FRAMES_TO_ANALYZE)
        if not frames:
            print("[ERRO] Nenhum frame capturado")
            return

        print(f"[diag] {len(frames)} frames capturados. Analisando...")

        # Analisa cada frame
        for frame_idx, frame in enumerate(frames):
            print(f"\n\n{'#'*70}")
            print(f"### FRAME {frame_idx+1}/{len(frames)} - Shape: {frame.shape}")
            print(f"{'#'*70}")

            # Salva frame completo
            full_path = os.path.join(OUTPUT_DIR, f"frame{frame_idx}_full.png")
            cv2.imwrite(full_path, frame)
            print(f"[diag] Frame completo salvo: {full_path}")

            # Analisa cada slot
            for slot_idx in range(3):
                analyze_slot(frame, slot_idx, frame_idx)

    finally:
        print("\n[diag] Encerrando scrcpy...")
        scrcpy_proc.terminate()
        try:
            scrcpy_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            scrcpy_proc.kill()

    print(f"\n\n{'='*70}")
    print("CONCLUSÃO DO DIAGNÓSTICO")
    print(f"{'='*70}")
    print(f"Arquivos salvos em: {OUTPUT_DIR}")
    print(f"Verifique os recortes (crop) e masks para inspeção visual.")


if __name__ == "__main__":
    main()
