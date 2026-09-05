#!/usr/bin/env python3
"""Single-event probe for scrcpy-server 4.1 Android injection."""

import os
import socket
import struct
import subprocess
import threading
import time

import cv2
import numpy as np

import adb_io
import board_catalog
import config
from vision import detect_board, read_pieces


SERVER = config.SCRCPY_SERVER_PATH
REMOTE = "/data/local/tmp/scrcpy-injection-probe.jar"
PORT = 27183
TYPE_TOUCH = 2
DOWN, UP, MOVE = 0, 1, 2


def adb(args, timeout=20):
    return subprocess.run(
        [config.ADB_PATH, "-s", config.ADB_DEVICE_ID] + args,
        capture_output=True,
        timeout=timeout,
    )


def recv_exact(sock, size):
    data = b""
    while len(data) < size:
        part = sock.recv(size - len(data))
        if not part:
            raise RuntimeError("control socket EOF")
        data += part
    return data


def metrics(frame, profile, slot):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 45)
    sx, _ = config.TRAY_SLOT_CENTERS[slot]
    tray = int(mask[1600:1900, sx - 145:sx + 145].sum())
    x1, y1 = profile["grid_top_left"]
    x2, y2 = profile["grid_bottom_right"]
    return tray, int(mask[y1:y2, x1:x2].sum())


def packet(action, pointer_id, x, y):
    return struct.pack(
        ">BBqiiHHHii", TYPE_TOUCH, action, pointer_id,
        int(x), int(y), 1080, 2340,
        0xFFFF if action != UP else 0, 0, 0,
    )


def run(pointer_id=-2):
    config.DEBUG = False
    profile = board_catalog.load_profile(11)
    before_frame = adb_io.capture_frame()
    before_board = detect_board(before_frame, profile=profile)
    pieces = read_pieces(before_frame)
    piece = next(p for p in pieces if p is not None and p.cells)
    source = (int(piece.pickup_cx), int(piece.pickup_cy))
    probe_move = (source[0] + 8, source[1] - 8)

    logcat_clear = adb(["logcat", "-c"], timeout=10)
    print(f"[SETUP] logcat_clear_rc={logcat_clear.returncode}", flush=True)

    scid = f"{int(time.time() * 1000) & 0xFFFFFFFF:08x}"
    name = f"scrcpy_{scid}"
    listener = socket.socket()
    server = control = None
    server_output = []
    packets = []
    try:
        adb(["push", SERVER, REMOTE], timeout=30)
        adb(["reverse", "--remove-all"], timeout=5)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", PORT))
        listener.listen(1)
        reverse = adb(["reverse", f"localabstract:{name}", f"tcp:{PORT}"], timeout=10)
        if reverse.returncode != 0:
            raise RuntimeError((reverse.stderr or b"").decode(errors="replace"))
        server = subprocess.Popen([
            config.ADB_PATH, "-s", config.ADB_DEVICE_ID, "shell",
            f"CLASSPATH={REMOTE}", "app_process", "/",
            "com.genymobile.scrcpy.Server", "4.1", f"scid={scid}",
            "log_level=debug", "video=false", "audio=false", "control=true",
            "power_on=false",
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
           text=True, bufsize=1)

        def drain_server():
            if server.stdout is not None:
                for line in server.stdout:
                    server_output.append(line.rstrip())

        threading.Thread(target=drain_server, daemon=True).start()
        listener.settimeout(12)
        control, _ = listener.accept()
        metadata = recv_exact(control, 64).rstrip(b"\0").decode(errors="replace")
        print(f"[HANDSHAKE] metadata={metadata}", flush=True)

        def send(action, x, y, label):
            raw = packet(action, pointer_id, x, y)
            timestamp = time.monotonic_ns()
            control.sendall(raw)
            packets.append({
                "label": label,
                "host_monotonic_ns": timestamp,
                "action": action,
                "pointer_id": pointer_id,
                "x": x,
                "y": y,
                "bytes_hex": raw.hex(),
            })

        send(DOWN, *source, "DOWN")
        time.sleep(0.50)
        send(MOVE, *probe_move, "MOVE")
        time.sleep(0.20)
        send(UP, *probe_move, "UP")
        print(f"[PACKETS] {packets}", flush=True)
        time.sleep(2.0)
    finally:
        if control is not None:
            control.close()
        listener.close()
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server.kill()
        adb(["reverse", "--remove", f"localabstract:{name}"], timeout=5)

    after_frame = adb_io.capture_frame()
    after_board = detect_board(after_frame, profile=profile)
    after_pieces = read_pieces(after_frame)
    before_tray, before_mass = metrics(before_frame, profile, piece.id)
    after_tray, after_mass = metrics(after_frame, profile, piece.id)
    board_changed = not np.array_equal(before_board.grid, after_board.grid)
    print(f"[VISUAL] tray={before_tray}->{after_tray} "
          f"board_mass={before_mass}->{after_mass} "
          f"board_changed={board_changed}", flush=True)
    print(f"[VISUAL] pieces_after={[(p.id, sorted(p.cells)) for p in after_pieces if p and p.cells]}", flush=True)
    relevant_logs = [line for line in server_output
                     if any(term in line.lower() for term in
                            ("inject", "control", "touch", "permission", "exception"))]
    print(f"[SERVER_LOG] {relevant_logs}", flush=True)
    print("[MATRIX] MotionEvent_created=YES (source path); "
          "injectInputEvent_return=NOT_OBSERVED; "
          "Android_accept=NOT_OBSERVED; "
          f"Unity_visual={'YES' if after_tray < before_tray or board_changed else 'NO'}", flush=True)


if __name__ == "__main__":
    run()
