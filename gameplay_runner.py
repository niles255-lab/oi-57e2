#!/usr/bin/env python3
"""Headless, fail-closed gameplay runner for real-device validation."""

from __future__ import annotations

import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import adb_io
import board_catalog
import bot
import config
from model import Board, Piece, Move, apply_move, is_legal
from solver import best_sequence
from vision import detect_board, read_pieces


PACKAGE = "com.fumbgames.bitcoinblockstm"
ACTIVITY = PACKAGE + "/com.unity3d.player.UnityPlayerActivity"
CONSENSUS_FRAMES = 3
CONSENSUS_POLL_S = 0.18
POST_SETTLE_S = 1.0


@dataclass
class ObservedState:
    frame: np.ndarray
    board: Board
    pieces: list
    profile: dict

    @property
    def signature(self):
        board_sig = (self.board.rows, self.board.cols,
                     self.board.grid.tobytes())
        piece_sig = tuple(
            (p.id, p.width, p.height, tuple(sorted(p.cells)))
            for p in self.pieces if p is not None and p.cells
        )
        return board_sig, piece_sig


def _cell_set(grid):
    return set(zip(*np.where(grid == 1)))


def _piece_signature(pieces, slot):
    for piece in pieces or []:
        if piece is not None and piece.id == slot and piece.cells:
            return piece.width, piece.height, tuple(sorted(piece.cells))
    return None


def _board_text(board):
    return "\n".join(
        "".join("#" if board.grid[r, c] else "." for c in range(board.cols))
        for r in range(board.rows)
    )


def _piece_text(piece):
    if piece is None or not piece.cells:
        return None
    return {
        "slot": piece.id,
        "shape": sorted(piece.cells),
        "size": (piece.width, piece.height),
        "blocks": len(piece.cells),
        "pickup": None if piece.pickup_cx is None else
                  (round(piece.pickup_cx, 1), round(piece.pickup_cy, 1)),
    }


def _metrics(frame, profile, slot):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 45)
    sx, _ = config.TRAY_SLOT_CENTERS[slot]
    tray = int(mask[1600:1900, max(0, sx - 145):min(1080, sx + 145)].sum())
    x1, y1 = profile["grid_top_left"]
    x2, y2 = profile["grid_bottom_right"]
    board = int(mask[y1:y2, x1:x2].sum())
    return tray, board


def _launch_game():
    return adb_io._cmd(["shell", "am", "start", "-n", ACTIVITY], timeout=15.0)


def _known_profile(frame):
    status, detected, profile = board_catalog.ensure_board(frame)
    if status != "known":
        return None, status, detected
    return profile, status, detected


def _read_state(profile=None) -> Optional[ObservedState]:
    frame = adb_io.capture_frame()
    if frame is None:
        return None
    if profile is None:
        profile, status, _ = _known_profile(frame)
        if profile is None:
            print(f"[DETECTION_FAILURE] board_catalog={status}", flush=True)
            return None
    board = detect_board(frame, profile=profile)
    pieces = read_pieces(frame)
    return ObservedState(frame, board, pieces, profile)


def _stable_state(profile=None, timeout=12.0):
    deadline = time.monotonic() + timeout
    last_sig = None
    count = 0
    stable = None
    while time.monotonic() < deadline:
        state = _read_state(profile)
        if state is None:
            time.sleep(CONSENSUS_POLL_S)
            continue
        if state.signature == last_sig:
            count += 1
        else:
            last_sig = state.signature
            count = 1
        stable = state
        if count >= CONSENSUS_FRAMES:
            return stable
        time.sleep(CONSENSUS_POLL_S)
    return None


def _sample_trace(stop, samples, profile, slot, started):
    while not stop.is_set():
        # Native ADB frames are slower but authoritative for gameplay state;
        # scrcpy remains a monitor/diagnostic source only.
        frame = adb_io.capture_frame()
        if frame is not None:
            tray, board = _metrics(frame, profile, slot)
            samples.append((time.monotonic() - started, tray, board))
        time.sleep(0.03)


def _segment(samples, start, end):
    return [sample for sample in samples if start <= sample[0] <= end]


def _segment_report(name, samples):
    if not samples:
        print(f"[TRACE] {name}=NO_SAMPLES", flush=True)
        return None
    trays = [s[1] for s in samples]
    boards = [s[2] for s in samples]
    result = {
        "n": len(samples),
        "tray_median": int(statistics.median(trays)),
        "tray_min": min(trays),
        "tray_max": max(trays),
        "board_median": int(statistics.median(boards)),
        "board_min": min(boards),
        "board_max": max(boards),
    }
    print(f"[TRACE] {name}={result}", flush=True)
    return result


def _expected_after(board, piece, move):
    expected, lines, cols = apply_move(board, piece, move.row, move.col)
    return expected, lines, cols


def _classify(executor_ok, before, after, piece, move, expected, trace):
    actual = _cell_set(after.board.grid)
    wanted = _cell_set(expected.grid)
    missing = sorted(wanted - actual)
    extra = sorted(actual - wanted)
    board_changed = not np.array_equal(before.board.grid, after.board.grid)
    removed = (_piece_signature(before.pieces, move.piece_index)
               != _piece_signature(after.pieces, move.piece_index))
    baseline = trace.get("before", {}).get("tray_median", 0)
    during_min = trace.get("during", {}).get("tray_min", baseline)
    pickup = bool(baseline and during_min < 0.70 * baseline)
    confirmed = board_changed and removed and not missing and not extra
    if confirmed:
        verdict = "EFFECTIVE"
    elif board_changed:
        verdict = "WRONG_PLACEMENT"
    elif removed:
        verdict = "DROP_FAILURE"
    elif pickup:
        verdict = "DROP_FAILURE"
    elif not executor_ok:
        verdict = "EXECUTOR_FAILURE"
    else:
        verdict = "PICKUP_FAILURE"
    return {
        "verdict": verdict,
        "expected_cells": sorted(wanted),
        "actual_cells": sorted(actual),
        "missing_cells": missing,
        "extra_cells": extra,
        "board_changed": board_changed,
        "piece_removed_from_tray": removed,
        "pickup_confirmed": pickup,
        "placement_confirmed": confirmed,
    }


def run(max_moves=6):
    config.DRAG_PRE_HOLD_MS = 500
    config.DEBUG = False
    if not adb_io.check_device():
        print("[EXECUTOR_FAILURE] device not connected", flush=True)
        return []

    launch = _launch_game()
    print(f"[GAME] launch_rc={launch.returncode} "
          f"{(launch.stdout or b'').decode(errors='replace').strip()}", flush=True)
    if not adb_io.init_scrcpy_stream():
        print("[EXECUTOR_FAILURE] scrcpy trace unavailable", flush=True)
        return []

    results = []
    profile = None
    try:
        for move_index in range(max_moves):
            state = _stable_state(profile, timeout=30.0)
            if state is None:
                results.append({"verdict": "DETECTION_FAILURE"})
                break
            profile = state.profile
            bot._BOARD_PROFILE = profile
            valid = [p for p in state.pieces if p is not None and p.cells]
            print(f"\n[GAME] turn={move_index + 1} board={state.board.rows}x{state.board.cols} "
                  f"fill={int(state.board.grid.sum())}", flush=True)
            print(f"[GAME] board_before=\n{_board_text(state.board)}", flush=True)
            print(f"[GAME] pieces_detected={[_piece_text(p) for p in state.pieces]}", flush=True)
            if not valid:
                print("[GAME] no playable pieces; stopping", flush=True)
                break

            sequence = best_sequence(state.board, state.pieces)
            print(f"[GAME] solver_sequence={[(m.piece_index, m.row, m.col) for m in sequence]}", flush=True)
            if not sequence:
                results.append({"verdict": "SOLVER_FAILURE"})
                break
            move = sequence[0]
            piece = next((p for p in valid if p.id == move.piece_index), None)
            if piece is None or not is_legal(state.board, piece, move.row, move.col):
                results.append({"verdict": "SOLVER_FAILURE"})
                break
            try:
                expected, cleared_rows, cleared_cols = _expected_after(state.board, piece, move)
            except ValueError:
                results.append({"verdict": "SOLVER_FAILURE"})
                break

            px = int(piece.pickup_cx) if piece.pickup_cx is not None else config.TRAY_SLOT_CENTERS[piece.id][0]
            py = int(piece.pickup_cy) if piece.pickup_cy is not None else config.TRAY_SLOT_CENTERS[piece.id][1]
            target = bot._target_pixel(move, piece, state.board)
            print(f"[EXEC] pickup=({px},{py}) target_cell=({move.row},{move.col}) "
                  f"target_pixel={target} pre_hold={config.DRAG_PRE_HOLD_MS} "
                  f"duration={config.DRAG_DURATION_MS}", flush=True)
            print(f"[VERIFY] expected_new={sorted(_cell_set(expected.grid) - _cell_set(state.board.grid))} "
                  f"cleared_rows={cleared_rows} cleared_cols={cleared_cols}", flush=True)

            samples = []
            stop = threading.Event()
            started = time.monotonic()
            sampler = threading.Thread(target=_sample_trace,
                                       args=(stop, samples, profile, piece.id, started),
                                       daemon=True)
            sampler.start()
            time.sleep(0.5)
            exec_start = time.monotonic()
            executor_ok = bool(bot.execute_move(move, piece,
                                                before_frame=state.frame,
                                                board=state.board))
            exec_end = time.monotonic()
            time.sleep(POST_SETTLE_S)
            stop.set()
            sampler.join(timeout=1.0)
            before_trace = _segment_report("before", _segment(samples, 0, exec_start - started - 0.05))
            during_trace = _segment_report("during", _segment(samples, exec_start - started, exec_end - started + 0.05))
            after = _stable_state(profile, timeout=30.0)
            if after is None:
                result = {"verdict": "VERIFICATION_FAILURE"}
                results.append(result)
                break
            trace = {"before": before_trace or {}, "during": during_trace or {}}
            result = _classify(executor_ok, state, after, piece, move, expected, trace)
            result.update({"executor_return": executor_ok,
                           "elapsed": round(exec_end - exec_start, 3)})
            print(f"[TRACE] tray_before={trace['before'].get('tray_median')} "
                  f"tray_during={trace['during'].get('tray_min')} "
                  f"board_before={trace['before'].get('board_median')} "
                  f"board_after={_metrics(after.frame, profile, piece.id)[1]}", flush=True)
            print(f"[VERIFY] {result}", flush=True)
            results.append(result)
            if result["verdict"] != "EFFECTIVE":
                break
        return results
    finally:
        adb_io.stop_scrcpy_stream()


if __name__ == "__main__":
    run()
