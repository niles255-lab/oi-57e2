#!/usr/bin/env python3
"""
Overlay transparente sobre o scrcpy - mostra diagnóstico sem bloquear o jogo.
"""

import sys
import time
import subprocess
import ctypes
import ctypes.wintypes
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

# Windows API constants
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_POPUP = 0x80000000
LWA_ALPHA = 0x00000002
LWA_COLORKEY = 0x00000001
ULW_ALPHA = 0x00000002
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01

# Windows API structures
class Rect(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

class Size(ctypes.Structure):
    _fields_ = [("cx", ctypes.c_long), ("cy", ctypes.c_long)]

class BlendFunction(ctypes.Structure):
    _fields_ = [
        ("BlendOp", ctypes.c_byte),
        ("BlendFlags", ctypes.c_byte),
        ("SourceConstantAlpha", ctypes.c_byte),
        ("AlphaFormat", ctypes.c_byte),
    ]


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


def create_transparent_overlay(x: int, y: int, width: int, height: int) -> int:
    """Cria janela overlay transparente com WS_EX_LAYERED | WS_EX_TRANSPARENT."""
    user32 = ctypes.windll.user32
    
    # Registra classe de janela
    class WNDCLASSEX(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_uint),
            ("style", ctypes.c_uint),
            ("lpfnWndProc", ctypes.c_void_p),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", ctypes.c_void_p),
            ("hIcon", ctypes.c_void_p),
            ("hCursor", ctypes.c_void_p),
            ("hbrBackground", ctypes.c_void_p),
            ("lpszMenuName", ctypes.c_wchar_p),
            ("lpszClassName", ctypes.c_wchar_p),
            ("hIconSm", ctypes.c_void_p),
        ]
    
    wndclass = WNDCLASSEX()
    wndclass.cbSize = ctypes.sizeof(WNDCLASSEX)
    wndclass.style = 0
    wndclass.lpfnWndProc = ctypes.c_void_p(0)  # DefWindowProc
    wndclass.cbClsExtra = 0
    wndclass.cbWndExtra = 0
    wndclass.hInstance = ctypes.windll.kernel32.GetModuleHandleW(None)
    wndclass.hIcon = 0
    wndclass.hCursor = user32.LoadCursorW(0, 32512)  # IDC_ARROW
    wndclass.hbrBackground = 0
    wndclass.lpszMenuName = None
    wndclass.lpszClassName = "TransparentOverlayClass"
    wndclass.hIconSm = 0
    
    atom = user32.RegisterClassExW(ctypes.byref(wndclass))
    if not atom:
        raise ctypes.WinError()
    
    # Cria janela com ex style: LAYERED | TRANSPARENT | TOPMOST
    ex_style = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST
    style = WS_POPUP
    
    hwnd = user32.CreateWindowExW(
        ex_style,
        "TransparentOverlayClass",
        "DiagnosticOverlay",
        style,
        x, y, width, height,
        0, 0,
        ctypes.windll.kernel32.GetModuleHandleW(None),
        None
    )
    
    if not hwnd:
        raise ctypes.WinError()
    
    # Configura alpha da janela inteira (255 = opaco, 0 = transparente)
    # Usamos LWA_ALPHA para permitir alpha per-pixel via UpdateLayeredWindow
    user32.SetLayeredWindowAttributes(hwnd, 0, 255, LWA_ALPHA)
    
    return hwnd


def update_layered_window(hwnd: int, bgra_image: np.ndarray, x: int, y: int, width: int, height: int):
    """Atualiza janela layered com imagem BGRA (com canal alpha)."""
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    
    # Obtém DC da tela
    hdc_screen = user32.GetDC(0)
    hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
    
    # Cria bitmap compatível
    bmi = ctypes.create_string_buffer(40)
    ctypes.memmove(bmi, ctypes.c_int(40).to_bytes(4, 'little'), 4)  # biSize
    ctypes.memmove(bmi[4:8], ctypes.c_int(width).to_bytes(4, 'little'), 4)  # biWidth
    ctypes.memmove(bmi[8:12], ctypes.c_int(-height).to_bytes(4, 'little'), 4)  # biHeight (negativo = top-down)
    ctypes.memmove(bmi[12:14], ctypes.c_short(1).to_bytes(2, 'little'), 2)  # biPlanes
    ctypes.memmove(bmi[14:16], ctypes.c_short(32).to_bytes(2, 'little'), 2)  # biBitCount
    ctypes.memmove(bmi[16:20], ctypes.c_int(0).to_bytes(4, 'little'), 4)  # biCompression
    
    hbitmap = gdi32.CreateDIBSection(hdc_screen, ctypes.cast(bmi, ctypes.POINTER(ctypes.c_void_p)), 0, None, 0, 0)
    if not hbitmap:
        user32.ReleaseDC(0, hdc_screen)
        gdi32.DeleteDC(hdc_mem)
        raise ctypes.WinError()
    
    # Copia dados da imagem BGRA para o bitmap
    hbitmap_old = gdi32.SelectObject(hdc_mem, hbitmap)
    
    # Copia pixel por pixel (mais lento mas seguro)
    # Para performance real, usaríamos ctypes.memmove mas precisamos do ponteiro do DIB
    # Por simplicidade, usamos UpdateLayeredWindow com HDC
    
    # Prepara BLENDFUNCTION
    blend = BlendFunction()
    blend.BlendOp = AC_SRC_OVER
    blend.BlendFlags = 0
    blend.SourceConstantAlpha = 255
    blend.AlphaFormat = AC_SRC_ALPHA
    
    pt_src = Point(0, 0)
    pt_dst = Point(x, y)
    sz = Size(width, height)
    
    result = user32.UpdateLayeredWindow(
        hwnd,
        hdc_screen,
        ctypes.byref(pt_dst),
        ctypes.byref(sz),
        hdc_mem,
        ctypes.byref(pt_src),
        0,
        ctypes.byref(blend),
        ULW_ALPHA
    )
    
    # Cleanup
    gdi32.SelectObject(hdc_mem, hbitmap_old)
    gdi32.DeleteObject(hbitmap)
    gdi32.DeleteDC(hdc_mem)
    user32.ReleaseDC(0, hdc_screen)
    
    return result


def update_layered_window_simple(hwnd: int, bgra_image: np.ndarray, x: int, y: int, width: int, height: int):
    """Versão simplificada usando numpy array diretamente via UpdateLayeredWindow."""
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    
    hdc_screen = user32.GetDC(0)
    hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
    
    # BITMAPINFOHEADER
    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", ctypes.c_uint32),
            ("biWidth", ctypes.c_int32),
            ("biHeight", ctypes.c_int32),
            ("biPlanes", ctypes.c_int16),
            ("biBitCount", ctypes.c_int16),
            ("biCompression", ctypes.c_uint32),
            ("biSizeImage", ctypes.c_uint32),
            ("biXPelsPerMeter", ctypes.c_int32),
            ("biYPelsPerMeter", ctypes.c_int32),
            ("biClrUsed", ctypes.c_uint32),
            ("biClrImportant", ctypes.c_uint32),
        ]
    
    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = width
    bmi.biHeight = -height  # negative = top-down DIB
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0  # BI_RGB
    
    # Ponteiro para dados da imagem
    pv_bits = ctypes.c_void_p()
    hbitmap = gdi32.CreateDIBSection(hdc_screen, ctypes.byref(bmi), 0, ctypes.byref(pv_bits), 0, 0)
    if not hbitmap:
        user32.ReleaseDC(0, hdc_screen)
        gdi32.DeleteDC(hdc_mem)
        raise ctypes.WinError()
    
    # Copia dados BGRA para o bitmap
    img_data = bgra_image.ctypes.data_as(ctypes.c_void_p)
    ctypes.memmove(pv_bits, img_data, width * height * 4)
    
    hbitmap_old = gdi32.SelectObject(hdc_mem, hbitmap)
    
    blend = BlendFunction()
    blend.BlendOp = AC_SRC_OVER
    blend.BlendFlags = 0
    blend.SourceConstantAlpha = 255
    blend.AlphaFormat = AC_SRC_ALPHA
    
    pt_src = Point(0, 0)
    pt_dst = Point(x, y)
    sz = Size(width, height)
    
    result = user32.UpdateLayeredWindow(
        hwnd,
        hdc_screen,
        ctypes.byref(pt_dst),
        ctypes.byref(sz),
        hdc_mem,
        ctypes.byref(pt_src),
        0,
        ctypes.byref(blend),
        ULW_ALPHA
    )
    
    gdi32.SelectObject(hdc_mem, hbitmap_old)
    gdi32.DeleteObject(hbitmap)
    gdi32.DeleteDC(hdc_mem)
    user32.ReleaseDC(0, hdc_screen)
    
    return result


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


def get_window_borders(hwnd: int) -> tuple:
    wx, wy, ww, wh = get_window_rect(hwnd)
    cx, cy, cw, ch = get_client_rect(hwnd)
    return cx - wx, cy - wy, (wx + ww) - (cx + cw), (wy + wh) - (cy + ch)


def get_window_rect(hwnd: int) -> tuple:
    user32 = ctypes.windll.user32
    rect = Rect()
    if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    return 0, 0, 0, 0


def set_window_pos(hwnd: int, x: int, y: int, width: int, height: int):
    user32 = ctypes.windll.user32
    user32.SetWindowPos(hwnd, 0, x, y, width, height, 0x0004 | 0x0010 | 0x0040)


def draw_diagnostic_overlay(frame_bgr: np.ndarray, board, pieces, cw: int, ch: int, fps: float) -> np.ndarray:
    """Desenha overlay diagnóstico em imagem BGRA com alpha."""
    # Cria imagem BGRA (transparente por padrão)
    overlay = np.zeros((ch, cw, 4), dtype=np.uint8)
    
    # Copia frame original para o canal RGB (opcional - comentado para transparência total)
    # overlay[:, :, :3] = cv2.resize(frame_bgr, (cw, ch))
    # overlay[:, :, 3] = 0  # totalmente transparente por padrão
    
    # Desenha elementos do diagnóstico
    
    # 1. Grade do tabuleiro
    if board.rows > 0 and board.cols > 0:
        cell_w = cw / board.cols
        cell_h = ch / board.rows
        
        # Linhas da grade
        for i in range(board.cols + 1):
            x = int(i * cell_w)
            cv2.line(overlay, (x, 0), (x, ch), (255, 255, 255, 100), 1)
        for i in range(board.rows + 1):
            y = int(i * cell_h)
            cv2.line(overlay, (0, y), (cw, y), (255, 255, 255, 100), 1)
        
        # Células ocupadas
        for r in range(board.rows):
            for c in range(board.cols):
                if board.grid[r, c]:
                    x1 = int(c * cell_w)
                    y1 = int(r * cell_h)
                    x2 = int((c + 1) * cell_w)
                    y2 = int((r + 1) * cell_h)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0, 80), -1)
    
    # 2. Regiões das 3 peças (bandeja)
    # Coordenadas aproximadas na escala do frame
    scale_x = cw / 1080
    scale_y = ch / 2340
    piece_centers = [(int(220 * scale_x), int(1764 * scale_y)),
                     (int(516 * scale_x), int(1764 * scale_y)),
                     (int(813 * scale_x), int(1764 * scale_y))]
    half = int(130 * scale_x)
    
    for i, (px, py) in enumerate(piece_centers):
        # Retângulo da peça
        cv2.rectangle(overlay, (px - half, py - half), (px + half, py + half), (255, 160, 0, 150), 2)
        
        # Label da peça
        if i < len(pieces) and pieces[i] and pieces[i].cells:
            p = pieces[i]
            label = f"S{i}: {p.width}x{p.h} [{len(p.cells)}b]"
            cv2.putText(overlay, label, (px - half + 5, py + half - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255, 200), 1, cv2.LINE_AA)
    
    # 3. Painel de status (canto superior esquerdo)
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 280, 160
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0, 180), -1)
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 255, 100), 1)
    
    # Texto do painel
    from vision import _block_mask
    lines = [
        "DIAGNOSTICO VISUAL",
        f"Grade: {board.rows}x{board.cols}",
        f"Tabuleiro: {int(board.grid.sum())}/{board.rows*board.cols}",
        f"FPS: {fps:.1f}",
    ]
    for i, p in enumerate(pieces):
        if p and p.cells:
            lines.append(f"  S{i}: {p.width}x{p.height} [{len(p.cells)}b]")
        else:
            lines.append(f"  S{i}: vazio")
    
    y = panel_y + 20
    for line in lines:
        cv2.putText(overlay, line, (panel_x + 10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    
    return overlay


def main():
    print("=" * 70)
    print("OVERLAY TRANSPARENTE - Diagnóstico sobre o scrcpy")
    print("=" * 70)
    
    # 1. Encontra scrcpy
    hwnd = find_scrcpy_window(SCRCPY_WINDOW_TITLE, timeout=3.0)
    if not hwnd:
        print("[ERRO] scrcpy não encontrado")
        return
    
    # Aguarda cliente pronto
    for _ in range(50):
        cx, cy, cw, ch = get_client_rect(hwnd)
        if cw >= 300 and ch >= 500:
            break
        time.sleep(0.1)
    
    # Cria overlay transparente
    cx, cy, cw, ch = get_client_rect(hwnd)
    d_left, d_top, d_right, d_bottom = get_window_borders(hwnd)
    
    # Overlay janela sem bordas, mesmo tamanho da área cliente
    overlay_hwnd = create_transparent_overlay(cx, cy, cw, ch)
    print(f"[overlay] HWND: 0x{overlay_hwnd:X} @ ({cx},{cy}) {cw}x{ch}")
    
    # Loop
    with mss.mss() as sct:
        start = time.time()
        frame_count = 0
        
        print("\nOverlay ativo. Pressione 'q' para sair (5s auto-close)...")
        
        while time.time() - start < 10:
            if not ctypes.windll.user32.IsWindow(hwnd):
                break
            
            cx, cy, cw, ch = get_client_rect(hwnd)
            if cw == 0 or ch == 0:
                time.sleep(0.01)
                continue
            
            # Reposiciona overlay se scrcpy moveu
            d_left, d_top, d_right, d_bottom = get_window_borders(hwnd)
            overlay_x = cx - d_left
            overlay_y = cy - d_top
            overlay_w = cw + d_left + d_right
            overlay_h = ch + d_top + d_bottom
            # Simplificado: posiciona janela overlay na área cliente
            ctypes.windll.user32.SetWindowPos(hwnd, 0, cx, cy, cw, ch, 0x0004 | 0x0010 | 0x0040)
            
            # Captura frame
            img = mss.mss().grab({"left": cx, "top": cy, "width": cw, "height": ch})
            frame = np.array(img)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Processa visão
            board = detect_board(frame_bgr)
            pieces = read_pieces(frame_bgr)
            
            # Cria overlay
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            overlay_bgra = draw_diagnostic_overlay(frame_bgr, board, pieces, cw, ch, fps)
            
            # Atualiza overlay transparente
            update_layered_window_simple(overlay_hwnd, overlay_bgra, cx, cy, cw, ch)
            
            frame_count += 1
            time.sleep(0.016)  # ~60 FPS
        
        ctypes.windll.user32.DestroyWindow(overlay_hwnd)
        print(f"\nOverlay fechado. Frames: {frame_count}")


if __name__ == "__main__":
    main()
