#!/usr/bin/env python3
"""
Overlay transparente completo - diagnóstico visual sobre o scrcpy.
Fundo preto = transparente, elementos desenhados em cores.
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

class Size(ctypes.Structure):
    _fields_ = [("cx", ctypes.c_long), ("cy", ctypes.c_long)]

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32), ("biHeight", ctypes.c_int32),
                ("biPlanes", ctypes.c_int16), ("biBitCount", ctypes.c_int16), ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
                ("biClrImportant", ctypes.c_uint32)]


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


class OverlayWindow:
    """Janela overlay transparente com color key (preto = transparente)."""

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.hwnd = None
        self.hdc_window = None
        self.hdc_mem = None
        self.hbitmap = None
        self.pv_bits = None
        self.img_buffer = None

    def create(self):
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        gdi32 = ctypes.windll.gdi32

        # Registra classe
        class WNDCLASSEX(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("style", ctypes.c_uint), ("lpfnWndProc", ctypes.c_void_p),
                        ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int), ("hInstance", ctypes.c_void_p),
                        ("hIcon", ctypes.c_void_p), ("hCursor", ctypes.c_void_p), ("hbrBackground", ctypes.c_void_p),
                        ("lpszMenuName", ctypes.c_wchar_p), ("lpszClassName", ctypes.c_wchar_p), ("hIconSm", ctypes.c_void_p)]

        wc = WNDCLASSEX()
        wc.cbSize = ctypes.sizeof(WNDCLASSEX)
        wc.lpfnWndProc = ctypes.cast(user32.DefWindowProcW, ctypes.c_void_p)
        wc.hInstance = ctypes.windll.kernel32.GetModuleHandleW(None)
        wc.hCursor = ctypes.windll.user32.LoadCursorW(0, 32512)
        wc.lpszClassName = "DiagnosticOverlayClass"

        user32.RegisterClassExW(ctypes.byref(wc))

        # Cria janela: LAYERED | TRANSPARENT | TOPMOST
        self.hwnd = ctypes.windll.user32.CreateWindowExW(
            0x00080000 | 0x00000020 | 0x00000008,  # LAYERED | TRANSPARENT | TOPMOST
            "DiagnosticOverlayClass", "DiagnosticOverlay", 0x80000000,  # WS_POPUP
            self.x, self.y, self.width, self.height, 0, 0, ctypes.windll.kernel32.GetModuleHandleW(None), None
        )

        if not self.hwnd:
            raise ctypes.WinError()

        # Color key: preto (0) = transparente
        ctypes.windll.user32.SetLayeredWindowAttributes(self.hwnd, 0x000000, 0, 0x00000001)  # LWA_COLORKEY

        ctypes.windll.user32.ShowWindow(self.hwnd, 5)  # SW_SHOW
        ctypes.windll.user32.UpdateWindow(self.hwnd)

        # Prepara DCs e bitmap para desenho
        self.hdc_window = ctypes.windll.user32.GetDC(self.hwnd)
        self.hdc_mem = ctypes.windll.gdi32.CreateCompatibleDC(self.hdc_window)

        # Cria DIB section para desenho offscreen
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32), ("biHeight", ctypes.c_int32),
                        ("biPlanes", ctypes.c_int16), ("biBitCount", ctypes.c_int16), ("biCompression", ctypes.c_uint32),
                        ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                        ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
                        ("biClrImportant", ctypes.c_uint32)]

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = self.width
        bmi.biHeight = -self.height  # top-down
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0  # BI_RGB

        self.pv_bits = ctypes.c_void_p()
        self.hbitmap = ctypes.windll.gdi32.CreateDIBSection(self.hdc_window, ctypes.byref(bmi), 0, ctypes.byref(self.pv_bits), 0, 0)

        if not self.hbitmap:
            raise ctypes.WinError()

        # Buffer numpy para desenho rápido
        self.img_buffer = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self.hbm_old = ctypes.windll.gdi32.SelectObject(self.hdc_mem, self.hbitmap)

        print(f"[Overlay] HWND: 0x{self.hwnd:X} | {self.width}x{self.height} @ ({self.x},{self.y})")

    def clear(self):
        """Limpa buffer (preto = transparente)."""
        self.img_buffer[:, :, :] = 0

    def draw_diagnostic(self, frame_bgr: np.ndarray, board, pieces, fps: float):
        """Desenha elementos do diagnóstico no buffer."""
        self.clear()

        cw, ch = self.width, self.height

        # 1. Grade do tabuleiro
        if board.rows > 0 and board.cols > 0:
            cell_w = cw / board.cols
            cell_h = ch / board.rows

            # Linhas da grade
            for i in range(board.cols + 1):
                x = int(i * cell_w)
                cv2.line(self.img_buffer, (x, 0), (x, ch), (255, 255, 255, 100), 1)
            for i in range(board.rows + 1):
                y = int(i * cell_h)
                cv2.line(self.img_buffer, (0, y), (cw, y), (255, 255, 255, 100), 1)

            # Células ocupadas (verde semi-transparente)
            for r in range(board.rows):
                for c in range(board.cols):
                    if board.grid[r, c]:
                        x1 = int(c * cell_w)
                        y1 = int(r * cell_h)
                        x2 = int((c + 1) * cell_w)
                        y2 = int((r + 1) * cell_h)
                        cv2.rectangle(self.img_buffer, (x1, y1), (x2, y2), (0, 255, 0, 80), -1)

        # 2. Regiões das 3 peças (bandeja) - laranja
        scale_x = cw / 1080
        scale_y = ch / 2340
        piece_centers = [(int(220 * scale_x), int(1764 * scale_y)),
                         (int(516 * scale_x), int(1764 * scale_y)),
                         (int(813 * scale_x), int(1764 * scale_y))]
        half = int(130 * scale_x)

        for i, (px, py) in enumerate(piece_centers):
            cv2.rectangle(self.img_buffer, (px - half, py - half), (px + half, py + half), (255, 160, 0, 150), 2)

            if i < len(pieces) and pieces[i] and pieces[i].cells:
                p = pieces[i]
                label = f"S{i}: {p.width}x{p.height} [{len(p.cells)}b]"
                cv2.putText(self.img_buffer, label, (px - half + 5, py + half - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255, 200), 1, cv2.LINE_AA)

        # 3. Painel de status (canto sup. esquerdo)
        panel_x, panel_y = 10, 10
        panel_w, panel_h = 280, 160
        cv2.rectangle(self.img_buffer, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0, 180), -1)
        cv2.rectangle(self.img_buffer, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 255, 100), 1)

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
            cv2.putText(self.img_buffer, line, (panel_x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255, 255), 1, cv2.LINE_AA)
            y += 22

    def present(self):
        """Copia buffer para DIB section e faz BitBlt para a janela."""
        # Copia buffer numpy para DIB section
        ctypes.memmove(self.pv_bits, self.img_buffer.ctypes.data, self.width * self.height * 4)

        # BitBlt: mem DC -> window DC
        ctypes.windll.gdi32.BitBlt(self.hdc_window, 0, 0, self.width, self.height, self.hdc_mem, 0, 0, 0x00CC0020)  # SRCCOPY

    def destroy(self):
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        if self.hbm_old and self.hdc_mem:
            ctypes.windll.gdi32.SelectObject(self.hdc_mem, self.hbm_old)
        if self.hdc_mem:
            ctypes.windll.gdi32.DeleteDC(self.hdc_mem)
        if self.hbitmap:
            ctypes.windll.gdi32.DeleteObject(self.hbitmap)
        if self.hdc_window and self.hwnd:
            ctypes.windll.user32.ReleaseDC(self.hwnd, self.hdc_window)
        if self.hwnd:
            ctypes.windll.user32.DestroyWindow(self.hwnd)


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


def find_scrcpy_window(title: str, timeout: float = 15.0) -> int:
    user32 = ctypes.windll.user32
    start = time.time()
    while time.time() - start < timeout:
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            return hwnd
        time.sleep(0.1)
    return 0


def main():
    print("=" * 70)
    print("OVERLAY TRANSPARENTE - Diagnóstico Visual Completo")
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

    # Cria overlay
    overlay = OverlayWindow(cx, cy, cw, ch)
    overlay.create()

    print(f"[Overlay] Ativo - {cw}x{ch} @ ({cx},{cy})")
    print("Overlay transparente ativo. Pressione 'q' para sair (15s auto-close)...")

    # Loop
    with mss.mss() as sct:
        start = time.time()
        frame_count = 0

        while time.time() - start < 15:
            if not ctypes.windll.user32.IsWindow(hwnd):
                print("[diag] scrcpy fechado")
                break

            # Obtém área cliente ATUAL do scrcpy
            cx, cy, cw, ch = get_client_rect(hwnd)
            if cw == 0 or ch == 0:
                time.sleep(0.01)
                continue

            monitor = {"left": cx, "top": cy, "width": cw, "height": ch}

            # Captura frame do scrcpy
            img = sct.grab(monitor)
            frame = np.array(img)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Processa visão
            board = detect_board(frame)
            pieces = read_pieces(frame)

            # Desenha diagnóstico no overlay
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            overlay.draw_diagnostic(frame, board, pieces, fps)
            overlay.present()

            frame_count += 1
            time.sleep(0.016)  # ~60 FPS

    print(f"\nOverlay finalizado. Frames: {frame_count}")


if __name__ == "__main__":
    frame_count = 0
    start_time = time.time()
    main()
