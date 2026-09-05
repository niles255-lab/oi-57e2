#!/usr/bin/env python3
"""
Janela de diagnóstico com área cliente EXATA sobreposta à área cliente do scrcpy.
"""

import sys
import time
import subprocess
import ctypes
import os
import cv2
import numpy as np
import mss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import adb_io
from vision import read_pieces, detect_board, draw_overlay

# Config
SCRCPY_SERIAL = config.ADB_DEVICE_ID
SCRCPY_WINDOW_TITLE = f"scrcpy_bot_{SCRCPY_SERIAL}"
SCRCPY_MAX_SIZE = "1080"

# Windows API structures
class Rect(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

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


def get_window_rect(hwnd: int) -> tuple:
    user32 = ctypes.windll.user32
    rect = Rect()
    if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    return 0, 0, 0, 0


def get_client_rect(hwnd: int) -> tuple:
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


def set_window_pos(hwnd: int, x: int, y: int, width: int, height: int):
    """Define posição e tamanho da JANELA COMPLETA via Windows API."""
    user32 = ctypes.windll.user32
    flags = 0x0004 | 0x0010 | 0x0040  # SWP_NOZORDER | SWP_NOACTIVATE | SWP_SHOWWINDOW
    user32.SetWindowPos(hwnd, 0, x, y, width, height, flags)


def get_window_borders(hwnd: int) -> tuple:
    """Retorna (left, top, right, bottom) das bordas da janela."""
    wx, wy, ww, wh = get_window_rect(hwnd)
    cx, cy, cw, ch = get_client_rect(hwnd)
    left = cx - wx
    top = cy - wy
    right = (wx + ww) - (cx + cw)
    bottom = (wy + wh) - (cy + ch)
    return left, top, right, bottom


def print_status(hwnd: int, diag_hwnd: int):
    wx, wy, ww, wh = get_window_rect(hwnd)
    cx, cy, cw, ch = get_client_rect(hwnd)
    dx, dy, dw, dh = get_window_rect(diag_hwnd)
    dcx, dcy, dcw, dch = get_client_rect(diag_hwnd)
    
    print(f"\n{'='*60}")
    print(f"SCRCPY: janela={ww}x{wh} @({wx},{wy}) | cliente={cw}x{ch} @({cx},{cy})")
    print(f"DIAG:   janela={dw}x{dh} @({dx},{dy}) | cliente={dcw}x{dch} @({dcx},{dcy})")
    print(f"Match cliente: {dcw==cw and dch==ch and dcx==cx and dcy==cy}")
    print(f"{'='*60}")


def main():
    global frame_count
    frame_count = 0
    start_time = time.time()
    
    print("=" * 70)
    print("DIAGNÓSTICO - Cliente sobre Cliente (overlay perfeito)")
    print("=" * 70)
    
    # 1. Encontra/inicia scrcpy
    hwnd = find_scrcpy_window(SCRCPY_WINDOW_TITLE, timeout=3.0)
    scrcpy_proc = None
    
    if not hwnd:
        print("[diag] Iniciando scrcpy...")
        scrcpy_path = config.SCRCPY_PATH
        cmd = [scrcpy_path, f"--window-title={SCRCPY_WINDOW_TITLE}", "--max-size=1080", f"--serial={SCRCPY_SERIAL}"]
        scrcpy_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(4)
        hwnd = find_scrcpy_window(SCRCPY_WINDOW_TITLE, timeout=10.0)
    
    if not hwnd:
        print("[ERRO] scrcpy não encontrado")
        return
    
    # Aguarda cliente pronto
    for _ in range(50):
        cx, cy, cw, ch = get_client_rect(hwnd)
        if cw >= 300 and ch >= 500:
            break
        time.sleep(0.1)
    
    # Cria janela de diagnóstico
    cv2.namedWindow("Diagnostico", cv2.WINDOW_NORMAL)
    
    # Obtém HWND da janela OpenCV
    user32 = ctypes.windll.user32
    diag_hwnd = user32.FindWindowW(None, "Diagnostico")
    if not diag_hwnd:
        print("[ERRO] Não achou HWND da janela Diagnostico")
        return
    
    # Loop principal
    with mss.mss() as sct:
        print("\nLoop iniciado. Pressione 'q' para sair.\n")
        
        while True:
            if not ctypes.windll.user32.IsWindow(hwnd):
                print("[diag] scrcpy fechado")
                break
            
            # Obtém área cliente ATUAL do scrcpy
            cx, cy, cw, ch = get_client_rect(hwnd)
            if cw == 0 or ch == 0:
                time.sleep(0.05)
                continue
            
            # Calcula onde posicionar a janela Diagnostico para que sua ÁREA CLIENTE
            # coincida exatamente com a área cliente do scrcpy
            d_left, d_top, d_right, d_bottom = get_window_borders(diag_hwnd)
            
            # Posição da janela Diagnostico para que sua área cliente comece em (cx, cy)
            diag_x = cx - d_left
            diag_y = cy - d_top
            diag_w = cw + d_left + d_right
            diag_h = ch + d_top + d_bottom
            
            # Posiciona janela Diagnostico
            set_window_pos(diag_hwnd, diag_x, diag_y, diag_w, diag_h)
            
            monitor = {"left": cx, "top": cy, "width": cw, "height": ch}
            
            # Captura frame da área cliente do scrcpy
            img = sct.grab(monitor)
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Processa visão
            board = detect_board(frame)
            pieces = read_pieces(frame)
            vis = draw_overlay(frame.copy(), board, pieces)
            
            # Info overlay
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            info = [
                f"SCRCPY cliente: {cw}x{ch} @({cx},{cy})",
                f"DIAG cliente:   {dcw}x{dch} @({dcx},{dcy})" if 'dcx' in locals() else "",
                f"FPS: {fps:.1f} Frames:{frame_count}",
                f"Board: {board.rows}x{board.cols} ({int(board.grid.sum())})",
            ]
            for i, p in enumerate(pieces):
                if p and p.cells:
                    info.append(f"  S{i}: {p.width}x{p.height}[{len(p.cells)}]")
                else:
                    info.append(f"  S{i}: vazio")
            
            y = 20
            for line in info:
                if line:
                    cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                    y += 22
            
            # Atualiza vars para debug
            dcx, dcy, dcw, dch = get_client_rect(diag_hwnd)
            
            # Mostra
            cv2.imshow("Diagnostico", vis)
            
            # Debug periódico
            if frame_count % 30 == 0:
                print_status(hwnd, diag_hwnd)
            
            frame_count += 1
            
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
    
    cv2.destroyAllWindows()
    print(f"\nFinalizado. Frames: {frame_count}")


if __name__ == "__main__":
    frame_count = 0
    start_time = time.time()
    main()
