#!/usr/bin/env python3
"""Isolated scrcpy 4.1 control-protocol probe.

This file intentionally does not call the autoplay loop or modify the
solver/vision/verifier. It sends one continuous control-socket gesture and
uses native ADB frames only as visual evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import json
import os
import socket
import struct
import statistics
import subprocess
import threading
import time
from typing import Optional

import numpy as np

from blocks_bot import adb_io, board_catalog, bot, config
from blocks_bot.model import Move, apply_move, is_legal
from blocks_bot.vision import detect_board, read_pieces


CONTROL_TYPE_TOUCH = 2
ACTION_DOWN = 0
ACTION_UP = 1
ACTION_MOVE = 2
POINTER_GENERIC_FINGER = -2
POINTER_VIRTUAL_FINGER = -3

SERVER_PATH = config.SCRCPY_SERVER_PATH
REMOTE_SERVER = "/data/local/tmp/scrcpy-transport-probe.jar"
REVERSE_PORT = 27183
SCREEN_W = 1080
SCREEN_H = 2340


def _adb(args, timeout=20):
    return subprocess.run(
        [config.ADB_PATH, "-s", config.ADB_DEVICE_ID] + args,
        capture_output=True,
        timeout=timeout,
    )


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    out = bytearray()
    while len(out) < size:
        part = sock.recv(size - len(out))
        if not part:
            raise RuntimeError("scrcpy control socket closed during metadata")
        out.extend(part)
    return bytes(out)


def _piece_sig(pieces, slot):
    for piece in pieces or []:
        if piece is not None and piece.id == slot and piece.cells:
            return piece.width, piece.height, tuple(sorted(piece.cells))
    return None


def _piece_signature(pieces, slot):
    return _piece_sig(pieces, slot)


def _metrics(frame, profile, slot):
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 45)
    sx, _ = config.TRAY_SLOT_CENTERS[slot]
    tray = int(mask[1600:1900, max(0, sx - 145):min(1080, sx + 145)].sum())
    x1, y1 = profile["grid_top_left"]
    x2, y2 = profile["grid_bottom_right"]
    board = int(mask[y1:y2, x1:x2].sum())
    return tray, board


def _read_state(profile):
    frame = adb_io.capture_frame()
    if frame is None:
        raise RuntimeError("native ADB frame unavailable")
    board = detect_board(frame, profile=profile)
    pieces = read_pieces(frame)
    return frame, board, pieces


def _find_move(board, pieces):
    for piece in pieces:
        if piece is None or not piece.cells:
            continue
        for row in range(board.rows - piece.height + 1):
            for col in range(board.cols - piece.width + 1):
                if is_legal(board, piece, row, col):
                    return piece, Move(piece.id, row, col)
    raise RuntimeError("no legal diagnostic move available")


def _packet(action, pointer_id, x, y, width=SCREEN_W, height=SCREEN_H):
    """scrcpy 4.1 ControlMessage touch packet, exactly 32 bytes."""
    return struct.pack(
        ">BBqiiHHHii",
        CONTROL_TYPE_TOUCH,
        action,
        pointer_id,
        int(x),
        int(y),
        int(width),
        int(height),
        0xFFFF if action != ACTION_UP else 0,
        0,
        0,
    )


def _launch_server(scid, listener):
    pushed = _adb(["push", SERVER_PATH, REMOTE_SERVER], timeout=30)
    if pushed.returncode != 0:
        raise RuntimeError((pushed.stderr or b"").decode(errors="replace"))
    _adb(["reverse", "--remove", f"localabstract:scrcpy_{scid}"], timeout=5)
    reverse = _adb([
        "reverse", f"localabstract:scrcpy_{scid}",
        f"tcp:{REVERSE_PORT}",
    ], timeout=10)
    if reverse.returncode != 0:
        raise RuntimeError((reverse.stderr or b"").decode(errors="replace"))

    return subprocess.Popen([
        config.ADB_PATH, "-s", config.ADB_DEVICE_ID, "shell",
        f"CLASSPATH={REMOTE_SERVER}", "app_process", "/",
        "com.genymobile.scrcpy.Server", "4.1", f"scid={scid}",
        "log_level=debug", "video=false", "audio=false", "control=true",
        "power_on=false",
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
       text=True, bufsize=1)


def run(pointer_id: int, hold_ms=500, duration_ms=2000, steps=40):
    config.DEBUG = False
    profile = board_catalog.load_profile(11)
    if profile is None:
        raise RuntimeError("11x11 profile missing")
    before_frame, before_board, before_pieces = _read_state(profile)
    tray_before_native, board_mass_before_native = _metrics(
        before_frame, profile, next(p.id for p in before_pieces
                                    if p is not None and p.cells))
    piece, move = _find_move(before_board, before_pieces)
    expected_board, _, _ = apply_move(before_board, piece, move.row, move.col)
    source = (
        int(piece.pickup_cx) if piece.pickup_cx is not None else config.TRAY_SLOT_CENTERS[piece.id][0],
        int(piece.pickup_cy) if piece.pickup_cy is not None else config.TRAY_SLOT_CENTERS[piece.id][1],
    )
    bot._BOARD_PROFILE = profile
    target = bot._target_pixel(move, piece, before_board)
    print(f"[STATE] board={before_board.rows}x{before_board.cols} "
          f"fill={int(before_board.grid.sum())} slot={piece.id} "
          f"shape={sorted(piece.cells)}", flush=True)
    print(f"[MOVE] target_cell=({move.row},{move.col}) source={source} "
          f"target={target} pointer_id={pointer_id}", flush=True)

    samples = []
    stop = threading.Event()
    sample_start = time.monotonic()

    def sample_loop():
        while not stop.is_set():
            try:
                frame = adb_io.capture_frame()
                if frame is not None:
                    tray, board_mass = _metrics(frame, profile, piece.id)
                    samples.append((time.monotonic() - sample_start,
                                    tray, board_mass))
            except Exception:
                pass
            time.sleep(0.15)

    sampler = threading.Thread(target=sample_loop, daemon=True)
    sampler.start()
    server = None
    listener = socket.socket()
    control = None
    scid = f"{int(time.time() * 1000) & 0xFFFFFFFF:08x}"
    packets = []
    server_log = []
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", REVERSE_PORT))
        listener.listen(1)
        server = _launch_server(scid, listener)

        def drain_server_log():
            if server.stdout is None:
                return
            for line in server.stdout:
                server_log.append(line.rstrip())

        threading.Thread(target=drain_server_log, daemon=True).start()
        listener.settimeout(12.0)
        control, _ = listener.accept()
        control.settimeout(5.0)
        metadata = _recv_exact(control, 64)
        print(f"[HANDSHAKE] metadata={metadata.rstrip(bytes([0])).decode(errors='replace')}", flush=True)

        def send(action, x, y, phase):
            packet = _packet(action, pointer_id, x, y)
            now = time.monotonic_ns()
            control.sendall(packet)
            packets.append({
                "host_monotonic_ns": now,
                "phase": phase,
                "type": packet[0],
                "action": action,
                "pointer_id": pointer_id,
                "x": int(x),
                "y": int(y),
                "screen_width": SCREEN_W,
                "screen_height": SCREEN_H,
                "pressure_u16": 0xFFFF if action != ACTION_UP else 0,
                "action_button": 0,
                "buttons": 0,
                "bytes_hex": packet.hex(),
            })

        execute_start = time.monotonic()
        send(ACTION_DOWN, *source, "DOWN")
        time.sleep(hold_ms / 1000.0)
        for i in range(1, steps + 1):
            time.sleep(duration_ms / 1000.0 / steps)
            fraction = i / steps
            x = round(source[0] + (target[0] - source[0]) * fraction)
            y = round(source[1] + (target[1] - source[1]) * fraction)
            send(ACTION_MOVE, x, y, f"MOVE_{i}")
        send(ACTION_UP, *target, "UP")
        execute_end = time.monotonic()
        print(f"[EXEC] elapsed={execute_end - execute_start:.3f}s "
              f"packets={len(packets)}", flush=True)
        time.sleep(4.0)
    finally:
        stop.set()
        sampler.join(timeout=1.0)
        if control is not None:
            control.close()
        listener.close()
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                server.kill()
        _adb(["reverse", "--remove", f"localabstract:scrcpy_{scid}"], timeout=5)

    after_frame, after_board, after_pieces = _read_state(profile)
    expected = set(zip(*np.where(expected_board.grid == 1)))
    actual = set(zip(*np.where(after_board.grid == 1)))
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    board_changed = not np.array_equal(before_board.grid, after_board.grid)
    piece_removed = _piece_signature(before_pieces, piece.id) != _piece_signature(after_pieces, piece.id)
    before_tray = [s[1] for s in samples if s[0] < hold_ms / 1000.0]
    during_tray = [s[1] for s in samples if s[0] >= 0.0]
    tray_before = (int(statistics.median(before_tray))
                   if before_tray else tray_before_native)
    tray_after = _metrics(after_frame, profile, piece.id)[0]
    if board_changed and not missing and not extra and piece_removed:
        classification = "EFFECTIVE"
    elif board_changed:
        classification = "WRONG_PLACEMENT"
    elif piece_removed:
        classification = "DROP_FAILURE"
    else:
        classification = "PICKUP_FAILURE"
    log = {
        "pointer_id": pointer_id,
        "source": source,
        "target": target,
        "target_cell": [move.row, move.col],
        "expected_cells": sorted(expected),
        "actual_cells": sorted(actual),
        "missing_cells": missing,
        "extra_cells": extra,
        "tray_before": tray_before,
        "tray_after": tray_after,
        "board_before": before_board.grid.tolist(),
        "board_after": after_board.grid.tolist(),
        "board_changed": board_changed,
        "piece_removed": piece_removed,
        "classification": classification,
        "packets": packets,
    }
    log_path = os.path.join(
        os.environ.get("TEMP", os.getcwd()),
        f"scrcpy_transport_{pointer_id}_{dt.datetime.now():%Y%m%d_%H%M%S}.json",
    )
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2, default=int)
    print(f"[RESULT] classification={classification} tray_before={tray_before} "
          f"tray_after={tray_after} board_changed={board_changed} "
          f"missing={missing} extra={extra}", flush=True)
    print(f"[LOG] {log_path} packets={len(packets)}", flush=True)
    for line in server_log:
        if "touch" in line.lower() or "control" in line.lower():
            print(f"[SERVER] {line}", flush=True)
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointer-id", type=int, default=POINTER_GENERIC_FINGER)
    args = parser.parse_args()
    run(args.pointer_id)
