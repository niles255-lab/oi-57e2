#!/usr/bin/env python3
"""
Teste isolado do pipeline scrcpy -> mss -> vision.py.read_pieces()
Nao altera nenhum arquivo do projeto principal.
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

# Adiciona pasta do projeto ao path para importar vision.py
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from vision import read_pieces
import config

# Configurações do teste
SCRCPY_WINDOW_TITLE = "scrcpy_capture_test_pipeline"
SCRCPY_SERIAL = config.ADB_DEVICE_ID
SCRCPY_MAX_SIZE = "1080"
TEST_DURATION_SECONDS = 10


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
    """Localiza a janela do scrcpy pelo título. Retorna HWND ou 0."""
    user32 = ctypes.windll.user32
    start = time.time()
    while time.time() - start < timeout:
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            return hwnd
        time.sleep(0.1)
    return 0


def get_client_rect(hwnd: int) -> tuple:
    """
    Retorna (left, top, width, height) da ÁREA CLIENTE (conteúdo de vídeo)
    convertida para coordenadas de tela.
    """
    user32 = ctypes.windll.user32
    
    # Obtém rect do cliente (relativo à janela)
    client_rect = Rect()
    if not user32.GetClientRect(hwnd, ctypes.byref(client_rect)):
        return 0, 0, 0, 0
    
    cw = client_rect.right - client_rect.left
    ch = client_rect.bottom - client_rect.top
    
    # Converte (0,0) do cliente para coordenadas de tela
    pt = Point(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return 0, 0, 0, 0
    
    return pt.x, pt.y, cw, ch


def wait_for_window_ready(hwnd: int, min_width: int = 300, min_height: int = 500, timeout: float = 10.0) -> bool:
    """Aguarda a área cliente da janela atingir tamanho mínimo razoável."""
    user32 = ctypes.windll.user32
    start = time.time()
    while time.time() - start < timeout:
        left, top, width, height = get_client_rect(hwnd)
        if width >= min_width and height >= min_height:
            print(f"[capture] Janela pronta: client area {width}x{height}")
            return True
        time.sleep(0.2)
    print(f"[capture] Timeout: janela não atingiu tamanho mínimo ({min_width}x{min_height})")
    return False


def capture_loop(hwnd: int, frame_queue: "threading.Queue", stop_event: threading.Event):
    """Thread de captura contínua via mss. Captura apenas a área cliente (vídeo)."""
    # Aguarda janela estar pronta
    if not wait_for_window_ready(hwnd):
        print("[capture] Janela não ficou pronta a tempo")
        return

    left, top, width, height = get_client_rect(hwnd)
    if width == 0 or height == 0:
        print("[capture] Área cliente inválida")
        return

    monitor = {"left": left, "top": top, "width": width, "height": height}
    print(f"[capture] Monitor (client area): {monitor}")

    with mss.mss() as sct:
        while not stop_event.is_set():
            try:
                img = sct.grab(monitor)
                # mss retorna BGRA, converte para BGR (OpenCV padrão)
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # Mantém apenas o frame mais recente
                try:
                    frame_queue.get_nowait()
                except Exception:
                    pass
                frame_queue.put(frame)
            except Exception as e:
                print(f"[capture] Erro na captura: {e}")
                time.sleep(0.01)


def main():
    print("=" * 60)
    print("TESTE PIPELINE: scrcpy -> mss -> vision.py.read_pieces()")
    print("=" * 60)

    # 1. Inicia scrcpy em background
    scrcpy_path = config.SCRCPY_PATH
    scrcpy_cmd = [
        scrcpy_path,
        f"--window-title={SCRCPY_WINDOW_TITLE}",
        f"--max-size={SCRCPY_MAX_SIZE}",
        f"--serial={SCRCPY_SERIAL}",
    ]
    print(f"[main] Iniciando scrcpy: {' '.join(scrcpy_cmd)}")

    scrcpy_proc = subprocess.Popen(
        scrcpy_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 2. Aguarda janela aparecer
    print("[main] Aguardando janela do scrcpy...")
    hwnd = find_scrcpy_window(SCRCPY_WINDOW_TITLE, timeout=15.0)
    if not hwnd:
        print("[ERRO] Janela do scrcpy não encontrada!")
        scrcpy_proc.terminate()
        return

    # Verifica área cliente (vídeo)
    left, top, width, height = get_client_rect(hwnd)
    print(f"[main] Janela encontrada: HWND={hwnd}, client area pos=({left},{top}), size={width}x{height}")

    if width < 300 or height < 500:
        print("[AVISO] Área cliente muito pequena, pode ser problema de DPI ou janela não pronta")

    # 3. Inicia thread de captura
    frame_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=capture_loop,
        args=(hwnd, frame_queue, stop_event),
        daemon=True,
    )
    capture_thread.start()

    # Aguarda primeiro frame
    print("[main] Aguardando primeiro frame...")
    try:
        first_frame = frame_queue.get(timeout=5.0)
        print(f"[main] Primeiro frame recebido: {first_frame.shape}")
    except Exception:
        print("[ERRO] Timeout esperando primeiro frame")
        stop_event.set()
        scrcpy_proc.terminate()
        return

    # 4. Loop principal de teste
    print(f"\n[main] Iniciando teste por {TEST_DURATION_SECONDS}s...")
    print("-" * 60)

    start_time = time.time()
    frames_captured = 0
    frames_processed = 0
    read_pieces_calls = 0
    read_pieces_errors = 0
    last_pieces = [None, None, None]

    try:
        while time.time() - start_time < TEST_DURATION_SECONDS:
            # Pega frame mais recente (non-blocking)
            try:
                frame = frame_queue.get_nowait()
                frames_captured += 1
            except Exception:
                time.sleep(0.001)
                continue

            # Chama read_pieces
            read_pieces_calls += 1
            try:
                pieces = read_pieces(frame)
                frames_processed += 1
                last_pieces = pieces
            except Exception as e:
                read_pieces_errors += 1
                print(f"[read_pieces ERRO] {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

            # Log periódico
            if frames_processed % 30 == 0 and frames_processed > 0:
                elapsed = time.time() - start_time
                fps = frames_processed / elapsed
                print(f"  [{elapsed:.1f}s] FPS processamento: {fps:.1f}, frames: {frames_processed}")

    except KeyboardInterrupt:
        print("\n[main] Interrompido pelo usuário")

    finally:
        stop_event.set()
        capture_thread.join(timeout=2.0)

        # Encerra scrcpy
        print("[main] Encerrando scrcpy...")
        scrcpy_proc.terminate()
        try:
            scrcpy_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            scrcpy_proc.kill()

    # 5. Relatório final
    elapsed = time.time() - start_time
    capture_fps = frames_captured / elapsed if elapsed > 0 else 0
    process_fps = frames_processed / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 60)
    print("RESULTADO DO TESTE")
    print("=" * 60)
    print(f"Duração do teste:     {elapsed:.1f}s")
    print(f"FPS captura (mss):    {capture_fps:.1f}")
    print(f"Frames capturados:    {frames_captured}")
    print(f"Frames processados:   {frames_processed}")
    print(f"Resolução capturada:  {width}x{height}")
    print(f"Chamadas read_pieces: {read_pieces_calls}")
    print(f"Erros read_pieces:    {read_pieces_errors}")
    print(f"FPS processamento:    {process_fps:.1f}")

    print("\nPeças detectadas (último frame):")
    for i, piece in enumerate(last_pieces):
        if piece and piece.cells:
            cells_str = " ".join(f"({r},{c})" for r, c in piece.cells)
            print(f"  slot {i}: {piece.width}x{piece.height} [{len(piece.cells)} blk] {cells_str}")
        else:
            print(f"  slot {i}: vazio/None")

    # Verifica se pipeline funcionou
    pipeline_ok = (
        frames_captured > 0
        and frames_processed > 0
        and read_pieces_errors < read_pieces_calls * 0.5  # menos de 50% erro
    )

    print("\n" + "=" * 60)
    if pipeline_ok:
        print("[OK] PIPELINE FUNCIONOU: mss -> read_pieces() OK")
    else:
        print("[FALHA] PIPELINE COM PROBLEMAS")
    print("=" * 60)


if __name__ == "__main__":
    main()
