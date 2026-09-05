#!/usr/bin/env python3
"""
Janela de diagnóstico que acompanha exatamente a área de vídeo do scrcpy.
Usa APIs Windows para obter posição/tamanho reais da janela e área cliente.
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
from vision import read_pieces, detect_board, draw_overlay, draw_move

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
    """Retorna (left, top, width, height) da janela completa."""
    user32 = ctypes.windll.user32
    rect = Rect()
    if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    return 0, 0, 0, 0


def get_client_rect(hwnd: int) -> tuple:
    """Retorna (left, top, width, height) da área cliente em coordenadas de TELA."""
    user32 = ctypes.windll.user32
    
    # 1. Obtém rect do cliente (relativo à janela)
    client_rect = Rect()
    if not user32.GetClientRect(hwnd, ctypes.byref(client_rect)):
        return 0, 0, 0, 0
    cw = client_rect.right - client_rect.left
    ch = client_rect.bottom - client_rect.top
    
    # 2. Converte (0,0) do cliente para coordenadas de tela
    pt = Point(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return 0, 0, 0, 0
    
    return pt.x, pt.y, cw, ch


def print_window_info(hwnd: int):
    """Imprime todas as informações da janela e área cliente."""
    print(f"\n{'='*60}")
    print(f"INFORMAÇÕES DA JANELA SCRCPY")
    print(f"{'='*60}")
    print(f"HWND: {hwnd} (0x{hwnd:X})")
    
    # Janela completa
    wx, wy, ww, wh = get_window_rect(hwnd)
    print(f"\nJANELA COMPLETA (com bordas/título):")
    print(f"  Posição: ({wx}, {wy})")
    print(f"  Tamanho: {ww}x{wh}")
    print(f"  Aspecto: {ww/wh:.3f}")
    
    # Área cliente (vídeo)
    cx, cy, cw, ch = get_client_rect(hwnd)
    print(f"\nÁREA CLIENTE (vídeo renderizado):")
    print(f"  Posição na tela: ({cx}, {cy})")
    print(f"  Tamanho: {cw}x{ch}")
    print(f"  Aspecto: {cw/ch:.3f}")
    
    # Bordas
    border_left = cx - wx
    border_top = cy - wy
    border_right = (wx + ww) - (cx + cw)
    border_bottom = (wy + wh) - (cy + ch)
    print(f"\nBORDAS:")
    print(f"  Esquerda: {border_left}px")
    print(f"  Topo: {border_top}px")
    print(f"  Direita: {border_right}px")
    print(f"  Baixo: {border_bottom}px")
    
    return cx, cy, cw, ch


def main():
    print("=" * 60)
    print("DIAGNÓSTICO VISUAL - Overlay na janela do scrcpy")
    print("=" * 60)
    
    # 1. Inicia scrcpy se não estiver rodando
    hwnd = find_scrcpy_window(SCRCPY_WINDOW_TITLE, timeout=3.0)
    scrcpy_proc = None
    
    if not hwnd:
        print("[diag] Janela não encontrada, iniciando scrcpy...")
        scrcpy_path = adb_io._find_scrcpy_exe() if hasattr(adb_io, '_find_scrcpy_exe') else None
        if not scrcpy_path:
            scrcpy_path = config.SCRCPY_PATH
        
        cmd = [scrcpy_path, f"--window-title={SCRCPY_WINDOW_TITLE}", f"--max-size={SCRCPY_MAX_SIZE}", f"--serial={SCRCPY_SERIAL}"]
        print(f"[diag] Iniciando: {' '.join(cmd)}")
        scrcpy_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(4)
        hwnd = find_scrcpy_window(SCRCPY_WINDOW_TITLE, timeout=10.0)
    
    if not hwnd:
        print("[ERRO] Não foi possível encontrar/iniciar janela do scrcpy")
        return
    
    # 2. Aguarda área cliente estar pronta
    print("[diag] Aguardando área cliente ficar pronta...")
    for _ in range(50):
        cx, cy, cw, ch = get_client_rect(hwnd)
        if cw >= 300 and ch >= 500:
            break
        time.sleep(0.1)
    
    # 3. Mostra informações completas
    cx, cy, cw, ch = print_window_info(hwnd)
    
    # 4. Configura monitor MSS na área cliente EXATA
    monitor = {"left": cx, "top": cy, "width": cw, "height": ch}
    print(f"\nREGIÃO MSS CONFIGURADA:")
    print(f"  {monitor}")
    
    # 5. Loop de diagnóstico visual
    print(f"\n{'='*60}")
    print("INICIANDO LOOP DE DIAGNÓSTICO VISUAL")
    print("Pressione 'q' na janela OpenCV para sair")
    print("=" * 60)
    
    cv2.namedWindow("Diagnostico - Overlay scrcpy", cv2.WINDOW_NORMAL)
    
    # Tenta posicionar janela de diagnóstico ao lado do scrcpy
    wx, wy, ww, wh = get_window_rect(hwnd)
    cv2.moveWindow("Diagnostico - Overlay scrcpy", wx + ww + 10, wy)
    
    with mss.mss() as sct:
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Verifica se janela do scrcpy ainda existe
            if not ctypes.windll.user32.IsWindow(hwnd):
                print("[diag] Janela do scrcpy fechada")
                break
            
            # Re-obtém posição/tamanho atuais (caso janela tenha movido/redimensionado)
            cx, cy, cw, ch = get_client_rect(hwnd)
            if cw == 0 or ch == 0:
                time.sleep(0.1)
                continue
            
            # Atualiza monitor se tamanho mudou
            monitor = {"left": cx, "top": cy, "width": cw, "height": ch}
            
            # Captura frame da área cliente EXATA
            img = sct.grab(monitor)
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            frame_count += 1
            
            # Processa visão
            board = detect_board(frame)
            pieces = read_pieces(frame)
            
            # Desenha overlay no frame capturado
            vis = draw_overlay(frame.copy(), board, pieces)
            
            # Info extra na imagem
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            info_lines = [
                f"HWND: 0x{hwnd:X}",
                f"Cliente: {cw}x{ch} @ ({cx},{cy})",
                f"Frame: {frame.shape[1]}x{frame.shape[0]}",
                f"Aspecto: {cw/ch:.3f}",
                f"FPS: {fps:.1f}",
                f"Frames: {frame_count}",
                f"Board: {board.rows}x{board.cols} ({int(board.grid.sum())} filled)",
            ]
            
            for i, piece in enumerate(pieces):
                if piece and piece.cells:
                    info_lines.append(f"  Slot {i}: {piece.width}x{piece.height} [{len(piece.cells)} blk]")
                else:
                    info_lines.append(f"  Slot {i}: vazio")
            
            y0 = 20
            for line in info_lines:
                cv2.putText(vis, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                y0 += 22
            
            # Mostra janela de diagnóstico
            cv2.imshow("Diagnostico - Overlay scrcpy", vis)
            
            # Redimensiona janela de diagnóstico para manter proporção do frame
            # (OpenCV mantém aspecto automaticamente com WINDOW_NORMAL)
            
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            
            # Log periódico
            if frame_count % 60 == 0:
                print(f"[{elapsed:.1f}s] FPS: {fps:.1f} | Frame: {frame.shape} | Cliente: {cw}x{ch}")
    
    cv2.destroyAllWindows()
    
    if scrcpy_proc:
        scrcpy_proc.terminate()
        try:
            scrcpy_proc.wait(timeout=2)
        except:
            scrcpy_proc.kill()
    
    print(f"\n[diag] Finalizado. Total frames: {frame_count}")


if __name__ == "__main__":
    main()
