#!/usr/bin/env python3
"""
Diagnostico SOMENTE-LEITURA da tela ATUAL do Redmi Note 8.
- Abre o pacote via ADB (sem toques, sem drags).
- Captura UMA imagem via adb_io.capture_frame() (ADB screencap).
- Roda SOMENTE: detect_grid_size + detect_board + read_pieces.
- NAO executa solver, execute_move, taps ou alteracoes de codigo.
"""

import sys
import os
import time
import subprocess
import datetime
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import adb_io
from vision import detect_grid_size, detect_board, read_pieces
from test_offset_limits import search as mirror_search
from vision import NATIVE_DEVICE_W, NATIVE_DEVICE_H

PKG = "com.fumbgames.bitcoinblockstm"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "logs", "diag_readonly")
os.makedirs(OUTDIR, exist_ok=True)


def main():
    print("== diagnostico somente-leitura ==", flush=True)

    # 1. dispositivo
    ok = adb_io.check_device()
    print(f"ADB device {config.ADB_DEVICE_ID}: {'conectado' if ok else 'NAO ENCONTRADO'}", flush=True)
    if not ok:
        return

    # 2. abrir o pacote (launcher; nenhum toque/drag)
    print(f"abrindo pacote {PKG} ...", flush=True)
    r = subprocess.run(
        [config.ADB_PATH, "-s", config.ADB_DEVICE_ID, "shell", "monkey",
         "-p", PKG, "-c", "android.intent.category.LAUNCHER", "1"],
        capture_output=True, text=True, timeout=20)
    print(f"monkey launcher rc={r.returncode} out={(r.stdout or '')[:120]}", flush=True)

    # 3. estabilizar
    print("aguardando estabilizacao (8s)...", flush=True)
    time.sleep(8.0)

    # 4. capturar UMA imagem (mecanismo existente; sem stream => screencap ADB)
    fr = adb_io.capture_frame()
    if fr is None:
        print("ERRO: captura falhou", flush=True)
        return
    h, w = fr.shape[:2]

    # 5. garantir normalizacao 1080x2340
    print(f"RESOLUTION: {w}x{h}", flush=True)
    assert (w, h) == (NATIVE_DEVICE_W, NATIVE_DEVICE_H), "frame fora do nativo!"

    # 8. salvar captura
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cap_path = os.path.join(OUTDIR, f"live_{ts}.png")
    cv2.imwrite(cap_path, fr)
    print(f"captura salva em: {cap_path}", flush=True)

    # 6. deteccao (somente leitura)
    rows, cols = detect_grid_size(fr)
    print(f"BOARD_SIZE: {rows}x{cols} quadrado={rows == cols}", flush=True)
    board = detect_board(fr)
    print(f"BOARD: {board.rows}x{board.cols} "
          f"filled={int(board.grid.sum())}/{board.rows * board.cols}", flush=True)
    for r in range(board.rows):
        print("  " + "".join("#" if board.grid[r, c] else "." for c in range(board.cols)),
              flush=True)

    pieces = read_pieces(fr)

    # espelho exato da busca (para pitch/dx/dy/score por slot)
    import numpy as np
    from vision import _build_signal_map
    sig = _build_signal_map(fr)
    sc = w / NATIVE_DEVICE_W
    zero_set = (-8.0, -5.0, -2.0, 0.0, 1.0, 4.0, 7.0)

    for i, p in enumerate(pieces):
        print(f"PECA {i}:", flush=True)
        if p and p.cells:
            print(f"  forma={sorted(p.cells)}", flush=True)
            print(f"  blocos={len(p.cells)}", flush=True)
        else:
            print("  forma=None (nao detectada)", flush=True)
            continue
        sxn, syn = config.TRAY_SLOT_CENTERS[i]
        sx, sy = sxn * sc, syn * (h / NATIVE_DEVICE_H)
        best, _ = mirror_search(sig, w, h, sx, sy, sc, -8.0, 8.0, zero_set)
        if best is None:
            print("  (sem vencedor no espelho)", flush=True)
            continue
        pitch, dx, dy, pre, pos, n, ms, s = best
        print(f"  pitch={pitch:.1f}", flush=True)
        print(f"  dx={dx:+.0f}", flush=True)
        print(f"  dy={dy:+.0f}", flush=True)
        print(f"  score={s:.4f}", flush=True)
        print(f"  pre-snap={pre} pos-snap={pos}", flush=True)

    with open(os.path.join(OUTDIR, f"live_{ts}.txt"), "w", encoding="utf-8") as f:
        f.write(f"RESOLUTION: {w}x{h}\n")
        f.write(f"BOARD_SIZE: {board.rows}x{board.cols}\n")
        f.write(str(board) + "\n")
        for i, p in enumerate(pieces):
            f.write(f"PECA {i}: " + (f"{p.width}x{p.height} blocos={len(p.cells)} "
                                     f"{sorted(p.cells)}\n" if p and p.cells else "None\n"))
    print("resumo salvo.", flush=True)


if __name__ == "__main__":
    main()
